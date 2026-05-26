#include "pjrt_plugin.h"

#include <dlfcn.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {

constexpr size_t kRequiredApiStructSize =
    PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Buffer_ToHostBuffer);

template <typename T, size_t StructSize>
T make_args() {
    T args{};
    args.struct_size = StructSize;
    args.extension_start = nullptr;
    return args;
}

#define MAKE_ARGS(type) make_args<type, type##_STRUCT_SIZE>()

std::string buffer_type_to_string(PJRT_Buffer_Type type) {
    switch (type) {
        case PJRT_Buffer_Type_F32: return "F32";
        case PJRT_Buffer_Type_F64: return "F64";
        case PJRT_Buffer_Type_S32: return "S32";
        case PJRT_Buffer_Type_S64: return "S64";
        default: return "<unsupported>";
    }
}

std::string named_value_to_string(const PJRT_NamedValue& value) {
    std::ostringstream os;
    os << std::string(value.name, value.name_size) << "=";
    switch (value.type) {
        case PJRT_NamedValue_kString:
            os << std::string(value.string_value, value.value_size);
            break;
        case PJRT_NamedValue_kInt64:
            os << value.int64_value;
            break;
        case PJRT_NamedValue_kInt64List:
            os << "[";
            for (size_t i = 0; i < value.value_size; ++i) {
                if (i) os << ",";
                os << value.int64_array_value[i];
            }
            os << "]";
            break;
        case PJRT_NamedValue_kFloat:
            os << value.float_value;
            break;
        case PJRT_NamedValue_kBool:
            os << (value.bool_value ? "true" : "false");
            break;
        default:
            os << "<unknown>";
            break;
    }
    return os.str();
}

}  // namespace

PjrtPlugin::PjrtPlugin(std::string plugin_path)
    : plugin_path_(std::move(plugin_path)) {}

PjrtPlugin::~PjrtPlugin() {
    try {
        destroy_client();
    } catch (...) {
    }
    // Intentionally do not dlclose the PJRT plugin in this short-lived probe.
    // With a newer vendored header than the installed plugin, unloading after a
    // partially initialized failure path proved crash-prone.
    handle_ = nullptr;
}

void PjrtPlugin::load() {
    handle_ = dlopen(plugin_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
        throw std::runtime_error("dlopen failed for " + plugin_path_ + ": " + dlerror());
    }

    dlerror();
    auto* sym = dlsym(handle_, "GetPjrtApi");
    const char* err = dlerror();
    if (err != nullptr || sym == nullptr) {
        throw std::runtime_error("dlsym(GetPjrtApi) failed: " + std::string(err ? err : "symbol is null"));
    }

    auto get_api = reinterpret_cast<GetPjrtApiFn>(sym);
    api_ = get_api();
    if (api_ == nullptr) {
        throw std::runtime_error("GetPjrtApi returned null");
    }
    if (api_->struct_size < kRequiredApiStructSize) {
        std::ostringstream os;
        os << "PJRT_Api struct is too small for the subset used by this probe: plugin="
           << api_->struct_size << " required=" << kRequiredApiStructSize
           << " full_header=" << PJRT_Api_STRUCT_SIZE
           << ". A closer-matched PJRT header may still be required for newer APIs.";
        throw std::runtime_error(os.str());
    }
}

void PjrtPlugin::initialize() {
    if (api_ == nullptr) throw std::runtime_error("PJRT API is not loaded");
    auto args = MAKE_ARGS(PJRT_Plugin_Initialize_Args);
    check(api_->PJRT_Plugin_Initialize(&args), "PJRT_Plugin_Initialize");
}

void PjrtPlugin::create_client() {
    if (api_ == nullptr) throw std::runtime_error("PJRT API is not loaded");
    if (client_ != nullptr) return;
    auto args = MAKE_ARGS(PJRT_Client_Create_Args);
    args.create_options = nullptr;
    args.num_options = 0;
    args.kv_get_callback = nullptr;
    args.kv_get_user_arg = nullptr;
    args.kv_put_callback = nullptr;
    args.kv_put_user_arg = nullptr;
    args.kv_try_get_callback = nullptr;
    args.kv_try_get_user_arg = nullptr;
    args.client = nullptr;
    check(api_->PJRT_Client_Create(&args), "PJRT_Client_Create");
    client_ = args.client;
    if (client_ == nullptr) throw std::runtime_error("PJRT_Client_Create returned null client");
}

void PjrtPlugin::destroy_client() {
    if (api_ != nullptr && client_ != nullptr) {
        auto args = MAKE_ARGS(PJRT_Client_Destroy_Args);
        args.client = client_;
        check(api_->PJRT_Client_Destroy(&args), "PJRT_Client_Destroy");
        client_ = nullptr;
    }
}

std::string PjrtPlugin::api_version_string() const {
    if (api_ == nullptr) return "<not loaded>";
    std::ostringstream os;
    os << api_->pjrt_api_version.major_version << "."
       << api_->pjrt_api_version.minor_version;
    return os.str();
}

std::string PjrtPlugin::platform_name() const {
    if (api_ == nullptr || client_ == nullptr) return "<no client>";
    auto args = MAKE_ARGS(PJRT_Client_PlatformName_Args);
    args.client = client_;
    check(api_->PJRT_Client_PlatformName(&args), "PJRT_Client_PlatformName");
    return view(args.platform_name, args.platform_name_size);
}

std::string PjrtPlugin::platform_version() const {
    if (api_ == nullptr || client_ == nullptr) return "<no client>";
    auto args = MAKE_ARGS(PJRT_Client_PlatformVersion_Args);
    args.client = client_;
    check(api_->PJRT_Client_PlatformVersion(&args), "PJRT_Client_PlatformVersion");
    return view(args.platform_version, args.platform_version_size);
}

std::vector<std::string> PjrtPlugin::plugin_attributes() const {
    std::vector<std::string> out;
    if (api_ == nullptr) return out;
    auto args = MAKE_ARGS(PJRT_Plugin_Attributes_Args);
    check(api_->PJRT_Plugin_Attributes(&args), "PJRT_Plugin_Attributes");
    for (size_t i = 0; i < args.num_attributes; ++i) {
        out.push_back(named_value_to_string(args.attributes[i]));
    }
    return out;
}

std::vector<std::string> PjrtPlugin::device_strings(bool addressable_only) const {
    std::vector<std::string> out;
    if (api_ == nullptr || client_ == nullptr) return out;

    PJRT_Device* const* devices = nullptr;
    size_t num_devices = 0;
    if (addressable_only) {
        auto args = MAKE_ARGS(PJRT_Client_AddressableDevices_Args);
        args.client = client_;
        check(api_->PJRT_Client_AddressableDevices(&args), "PJRT_Client_AddressableDevices");
        devices = args.addressable_devices;
        num_devices = args.num_addressable_devices;
    } else {
        auto args = MAKE_ARGS(PJRT_Client_Devices_Args);
        args.client = client_;
        check(api_->PJRT_Client_Devices(&args), "PJRT_Client_Devices");
        devices = args.devices;
        num_devices = args.num_devices;
    }

    for (size_t i = 0; i < num_devices; ++i) {
        auto desc_args = MAKE_ARGS(PJRT_Device_GetDescription_Args);
        desc_args.device = devices[i];
        check(api_->PJRT_Device_GetDescription(&desc_args), "PJRT_Device_GetDescription");

        auto id_args = MAKE_ARGS(PJRT_DeviceDescription_Id_Args);
        id_args.device_description = desc_args.device_description;
        check(api_->PJRT_DeviceDescription_Id(&id_args), "PJRT_DeviceDescription_Id");

        auto kind_args = MAKE_ARGS(PJRT_DeviceDescription_Kind_Args);
        kind_args.device_description = desc_args.device_description;
        check(api_->PJRT_DeviceDescription_Kind(&kind_args), "PJRT_DeviceDescription_Kind");

        auto debug_args = MAKE_ARGS(PJRT_DeviceDescription_DebugString_Args);
        debug_args.device_description = desc_args.device_description;
        check(api_->PJRT_DeviceDescription_DebugString(&debug_args), "PJRT_DeviceDescription_DebugString");

        std::ostringstream os;
        os << "id=" << id_args.id
           << " kind=" << view(kind_args.device_kind, kind_args.device_kind_size)
           << " debug=" << view(debug_args.debug_string, debug_args.debug_string_size);
        out.push_back(os.str());
    }
    return out;
}

PJRT_Device* PjrtPlugin::first_addressable_device() const {
    if (api_ == nullptr || client_ == nullptr) {
        throw std::runtime_error("PJRT client is not initialized");
    }
    auto args = MAKE_ARGS(PJRT_Client_AddressableDevices_Args);
    args.client = client_;
    check(api_->PJRT_Client_AddressableDevices(&args), "PJRT_Client_AddressableDevices");
    if (args.num_addressable_devices == 0 || args.addressable_devices == nullptr) {
        throw std::runtime_error("PJRT client reported no addressable devices");
    }
    return args.addressable_devices[0];
}

PjrtExecutionResult PjrtPlugin::compile_and_execute_mlir(const std::string& mlir,
                                                        const std::string& compile_options,
                                                        const std::vector<float>& input_values,
                                                        const std::vector<int64_t>& input_dims) const {
    if (api_ == nullptr || client_ == nullptr) {
        throw std::runtime_error("PJRT client is not initialized");
    }
    if (input_dims.empty()) {
        throw std::runtime_error("input_dims must not be empty");
    }

    size_t expected_elements = 1;
    for (int64_t dim : input_dims) {
        if (dim <= 0) throw std::runtime_error("input_dims must be positive");
        expected_elements *= static_cast<size_t>(dim);
    }
    if (input_values.size() != expected_elements) {
        std::ostringstream os;
        os << "input_values size mismatch: got " << input_values.size()
           << " expected " << expected_elements;
        throw std::runtime_error(os.str());
    }

    PJRT_LoadedExecutable* loaded = nullptr;
    PJRT_Executable* executable = nullptr;
    PJRT_Buffer* input_buffer = nullptr;
    PJRT_Event* input_done = nullptr;
    PJRT_Event* execute_done = nullptr;
    std::vector<PJRT_Buffer*> output_buffers;

    auto cleanup = [&]() {
        for (PJRT_Buffer*& buffer : output_buffers) destroy_buffer(buffer, "PJRT_Buffer_Destroy(output)");
        destroy_buffer(input_buffer, "PJRT_Buffer_Destroy(input)");
        await_and_destroy_event(execute_done, "PJRT_Event_Await(execute)", "PJRT_Event_Destroy(execute)");
        await_and_destroy_event(input_done, "PJRT_Event_Await(input transfer)", "PJRT_Event_Destroy(input transfer)");
        destroy_executable(executable, "PJRT_Executable_Destroy");
        destroy_loaded_executable(loaded, "PJRT_LoadedExecutable_Destroy");
    };

    try {
        PJRT_Program program{};
        program.struct_size = PJRT_Program_STRUCT_SIZE;
        program.extension_start = nullptr;
        program.code = const_cast<char*>(mlir.data());
        program.code_size = mlir.size();
        program.format = "mlir";
        program.format_size = std::strlen(program.format);

        auto compile_args = MAKE_ARGS(PJRT_Client_Compile_Args);
        compile_args.client = client_;
        compile_args.program = &program;
        compile_args.compile_options = compile_options.empty() ? nullptr : compile_options.data();
        compile_args.compile_options_size = compile_options.size();
        compile_args.executable = nullptr;
        check(api_->PJRT_Client_Compile(&compile_args), "PJRT_Client_Compile");
        loaded = compile_args.executable;
        if (loaded == nullptr) throw std::runtime_error("PJRT_Client_Compile returned null executable");

        auto get_exec_args = MAKE_ARGS(PJRT_LoadedExecutable_GetExecutable_Args);
        get_exec_args.loaded_executable = loaded;
        check(api_->PJRT_LoadedExecutable_GetExecutable(&get_exec_args), "PJRT_LoadedExecutable_GetExecutable");
        executable = get_exec_args.executable;
        if (executable == nullptr) throw std::runtime_error("PJRT_LoadedExecutable_GetExecutable returned null executable");

        PjrtExecutionResult result;

        auto name_args = MAKE_ARGS(PJRT_Executable_Name_Args);
        name_args.executable = executable;
        check(api_->PJRT_Executable_Name(&name_args), "PJRT_Executable_Name");
        result.executable_name = view(name_args.executable_name, name_args.executable_name_size);

        auto outputs_args = MAKE_ARGS(PJRT_Executable_NumOutputs_Args);
        outputs_args.executable = executable;
        check(api_->PJRT_Executable_NumOutputs(&outputs_args), "PJRT_Executable_NumOutputs");
        if (outputs_args.num_outputs == 0) {
            throw std::runtime_error("compiled executable reported zero outputs");
        }

        auto upload_args = MAKE_ARGS(PJRT_Client_BufferFromHostBuffer_Args);
        upload_args.client = client_;
        upload_args.data = input_values.data();
        upload_args.type = PJRT_Buffer_Type_F32;
        upload_args.dims = input_dims.data();
        upload_args.num_dims = input_dims.size();
        upload_args.byte_strides = nullptr;
        upload_args.num_byte_strides = 0;
        upload_args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
        upload_args.device = first_addressable_device();
        upload_args.memory = nullptr;
        upload_args.device_layout = nullptr;
        upload_args.done_with_host_buffer = nullptr;
        upload_args.buffer = nullptr;
        check(api_->PJRT_Client_BufferFromHostBuffer(&upload_args), "PJRT_Client_BufferFromHostBuffer");
        input_done = upload_args.done_with_host_buffer;
        input_buffer = upload_args.buffer;
        if (input_buffer == nullptr) throw std::runtime_error("PJRT_Client_BufferFromHostBuffer returned null buffer");
        await_and_destroy_event(input_done, "PJRT_Event_Await(input transfer)", "PJRT_Event_Destroy(input transfer)");

        auto execute_options = MAKE_ARGS(PJRT_ExecuteOptions);
        execute_options.send_callbacks = nullptr;
        execute_options.recv_callbacks = nullptr;
        execute_options.num_send_ops = 0;
        execute_options.num_recv_ops = 0;
        execute_options.launch_id = 0;
        execute_options.non_donatable_input_indices = nullptr;
        execute_options.num_non_donatable_input_indices = 0;
        execute_options.context = nullptr;
        execute_options.call_location = "fennix_pjrt_probe";
        execute_options.num_tasks = 0;
        execute_options.task_ids = nullptr;
        execute_options.incarnation_ids = nullptr;
        execute_options.multi_slice_config = nullptr;
        execute_options.use_major_to_minor_data_layout_for_callbacks = true;

        PJRT_Buffer* argument_list[1] = {input_buffer};
        PJRT_Buffer* const* argument_lists[1] = {argument_list};
        output_buffers.assign(outputs_args.num_outputs, nullptr);
        PJRT_Buffer** output_lists[1] = {output_buffers.data()};
        PJRT_Event* execute_events[1] = {nullptr};

        auto exec_args = MAKE_ARGS(PJRT_LoadedExecutable_Execute_Args);
        exec_args.executable = loaded;
        exec_args.options = &execute_options;
        exec_args.argument_lists = argument_lists;
        exec_args.num_devices = 1;
        exec_args.num_args = 1;
        exec_args.output_lists = output_lists;
        exec_args.device_complete_events = execute_events;
        exec_args.execute_device = upload_args.device;
        check(api_->PJRT_LoadedExecutable_Execute(&exec_args), "PJRT_LoadedExecutable_Execute");
        execute_done = execute_events[0];
        await_and_destroy_event(execute_done, "PJRT_Event_Await(execute)", "PJRT_Event_Destroy(execute)");

        result.outputs.reserve(output_buffers.size());
        for (PJRT_Buffer* buffer : output_buffers) {
            if (buffer == nullptr) {
                throw std::runtime_error("PJRT_LoadedExecutable_Execute returned a null output buffer");
            }
            result.outputs.push_back(copy_buffer_to_host(buffer));
        }
        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

void PjrtPlugin::check(PJRT_Error* error, const char* context) const {
    if (error == nullptr) return;
    std::string message = error_message(error);
    if (api_ != nullptr && api_->PJRT_Error_Destroy != nullptr) {
        auto args = MAKE_ARGS(PJRT_Error_Destroy_Args);
        args.error = error;
        api_->PJRT_Error_Destroy(&args);
    }
    throw std::runtime_error(std::string(context) + " failed: " + message);
}

void PjrtPlugin::await_and_destroy_event(PJRT_Event*& event, const char* await_context, const char* destroy_context) const {
    if (event == nullptr) return;
    if (api_ != nullptr && api_->PJRT_Event_Await != nullptr) {
        auto await_args = MAKE_ARGS(PJRT_Event_Await_Args);
        await_args.event = event;
        check(api_->PJRT_Event_Await(&await_args), await_context);
    }
    if (api_ != nullptr && api_->PJRT_Event_Destroy != nullptr) {
        auto destroy_args = MAKE_ARGS(PJRT_Event_Destroy_Args);
        destroy_args.event = event;
        check(api_->PJRT_Event_Destroy(&destroy_args), destroy_context);
    }
    event = nullptr;
}

void PjrtPlugin::destroy_buffer(PJRT_Buffer*& buffer, const char* context) const {
    if (buffer == nullptr || api_ == nullptr || api_->PJRT_Buffer_Destroy == nullptr) return;
    auto args = MAKE_ARGS(PJRT_Buffer_Destroy_Args);
    args.buffer = buffer;
    check(api_->PJRT_Buffer_Destroy(&args), context);
    buffer = nullptr;
}

void PjrtPlugin::destroy_loaded_executable(PJRT_LoadedExecutable*& executable, const char* context) const {
    if (executable == nullptr || api_ == nullptr || api_->PJRT_LoadedExecutable_Destroy == nullptr) return;
    auto args = MAKE_ARGS(PJRT_LoadedExecutable_Destroy_Args);
    args.executable = executable;
    check(api_->PJRT_LoadedExecutable_Destroy(&args), context);
    executable = nullptr;
}

void PjrtPlugin::destroy_executable(PJRT_Executable*& executable, const char* context) const {
    if (executable == nullptr || api_ == nullptr || api_->PJRT_Executable_Destroy == nullptr) return;
    auto args = MAKE_ARGS(PJRT_Executable_Destroy_Args);
    args.executable = executable;
    check(api_->PJRT_Executable_Destroy(&args), context);
    executable = nullptr;
}

PjrtBufferData PjrtPlugin::copy_buffer_to_host(PJRT_Buffer* buffer) const {
    auto type_args = MAKE_ARGS(PJRT_Buffer_ElementType_Args);
    type_args.buffer = buffer;
    check(api_->PJRT_Buffer_ElementType(&type_args), "PJRT_Buffer_ElementType");

    auto dims_args = MAKE_ARGS(PJRT_Buffer_Dimensions_Args);
    dims_args.buffer = buffer;
    check(api_->PJRT_Buffer_Dimensions(&dims_args), "PJRT_Buffer_Dimensions");

    PjrtBufferData out;
    out.type = type_args.type;
    out.dims.assign(dims_args.dims, dims_args.dims + dims_args.num_dims);
    if (out.type != PJRT_Buffer_Type_F32) {
        throw std::runtime_error("Only F32 outputs are currently supported; got " + buffer_type_to_string(out.type));
    }

    auto size_args = MAKE_ARGS(PJRT_Buffer_ToHostBuffer_Args);
    size_args.src = buffer;
    size_args.host_layout = nullptr;
    size_args.dst = nullptr;
    size_args.dst_size = 0;
    size_args.event = nullptr;
    check(api_->PJRT_Buffer_ToHostBuffer(&size_args), "PJRT_Buffer_ToHostBuffer(size query)");
    await_and_destroy_event(size_args.event, "PJRT_Event_Await(output size query)", "PJRT_Event_Destroy(output size query)");

    std::vector<char> host_bytes(size_args.dst_size);
    auto copy_args = MAKE_ARGS(PJRT_Buffer_ToHostBuffer_Args);
    copy_args.src = buffer;
    copy_args.host_layout = nullptr;
    copy_args.dst = host_bytes.data();
    copy_args.dst_size = host_bytes.size();
    copy_args.event = nullptr;
    check(api_->PJRT_Buffer_ToHostBuffer(&copy_args), "PJRT_Buffer_ToHostBuffer(copy)");
    await_and_destroy_event(copy_args.event, "PJRT_Event_Await(output copy)", "PJRT_Event_Destroy(output copy)");

    if (host_bytes.size() % sizeof(float) != 0) {
        throw std::runtime_error("Unexpected F32 output byte size: " + std::to_string(host_bytes.size()));
    }
    out.f32_values.resize(host_bytes.size() / sizeof(float));
    std::memcpy(out.f32_values.data(), host_bytes.data(), host_bytes.size());
    return out;
}

std::string PjrtPlugin::error_message(PJRT_Error* error) const {
    if (api_ == nullptr || api_->PJRT_Error_Message == nullptr) {
        return "<PJRT error; no message function available>";
    }
    auto args = MAKE_ARGS(PJRT_Error_Message_Args);
    args.error = error;
    api_->PJRT_Error_Message(&args);
    return view(args.message, args.message_size);
}

std::string PjrtPlugin::view(const char* data, size_t size) {
    if (data == nullptr) return "";
    return std::string(data, size);
}

#undef MAKE_ARGS


#pragma once

#include <string>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

struct PjrtBufferData {
    PJRT_Buffer_Type type = PJRT_Buffer_Type_INVALID;
    std::vector<int64_t> dims;
    std::vector<float> f32_values;
};

struct PjrtExecutionResult {
    std::string executable_name;
    std::vector<PjrtBufferData> outputs;
};

struct PjrtExecutableHandle {
    PJRT_LoadedExecutable* loaded = nullptr;
    PJRT_Executable* executable = nullptr;
    std::string executable_name;
    size_t num_outputs = 0;
};

class PjrtPlugin {
public:
    explicit PjrtPlugin(std::string plugin_path);
    ~PjrtPlugin();

    PjrtPlugin(const PjrtPlugin&) = delete;
    PjrtPlugin& operator=(const PjrtPlugin&) = delete;

    void load();
    void initialize();
    void create_client();
    void destroy_client();

    const PJRT_Api* api() const { return api_; }
    PJRT_Client* client() const { return client_; }

    std::string api_version_string() const;
    std::string platform_name() const;
    std::string platform_version() const;
    std::vector<std::string> device_strings(bool addressable_only = true) const;
    std::vector<std::string> plugin_attributes() const;
    PJRT_Device* first_addressable_device() const;
    PjrtExecutableHandle compile_mlir(const std::string& mlir,
                                      const std::string& compile_options) const;
    PjrtExecutionResult execute_compiled(const PjrtExecutableHandle& executable,
                                         const std::vector<float>& input_values,
                                         const std::vector<int64_t>& input_dims) const;
    void destroy_compiled(PjrtExecutableHandle& executable) const;
    PjrtExecutionResult compile_and_execute_mlir(const std::string& mlir,
                                                 const std::string& compile_options,
                                                 const std::vector<float>& input_values,
                                                 const std::vector<int64_t>& input_dims) const;

private:
    using GetPjrtApiFn = const PJRT_Api* (*)();

    void check(PJRT_Error* error, const char* context) const;
    void await_and_destroy_event(PJRT_Event*& event, const char* await_context, const char* destroy_context) const;
    void destroy_buffer(PJRT_Buffer*& buffer, const char* context) const;
    void destroy_loaded_executable(PJRT_LoadedExecutable*& executable, const char* context) const;
    void destroy_executable(PJRT_Executable*& executable, const char* context) const;
    PjrtBufferData copy_buffer_to_host(PJRT_Buffer* buffer) const;
    std::string error_message(PJRT_Error* error) const;
    static std::string view(const char* data, size_t size);

    std::string plugin_path_;
    void* handle_ = nullptr;
    const PJRT_Api* api_ = nullptr;
    PJRT_Client* client_ = nullptr;
};


#include <chrono>
#include <CL/opencl.h>
#include <CL/cl_platform.h>
#include <iostream>

const char* g_pcsz_source =
"__kernel void memset(__global int * result, __global int* a, __global int * x, __global int * y, __global int * sz) \n"
"{ \n"
" int i = get_global_id(0); \n"
" if (i >= sz[0]) \n"
" { \n"
"  return;\n"
" } \n"
" result[i] = 0; \n"
" for(int k = 0; k < sz[0]; k++) \n"
" { \n"
"  result[i] = result[i] + a[0] * y[i * sz[0] + k] * x[k]; \n"
" } \n"
"} \n";

using namespace std::chrono;

int main()
{
	// 1. Получение платформы
	cl_uint u_num_platforms;
	clGetPlatformIDs(0, nullptr, &u_num_platforms);
	std::cout << u_num_platforms << " platforms" << std::endl;
	auto* p_platforms = new cl_platform_id[u_num_platforms];
	clGetPlatformIDs(u_num_platforms, p_platforms, &u_num_platforms);

	// 2. Получение информации о платформе
	constexpr size_t size = 128;
	char param_value[size] = { 0 };
	size_t param_value_size_ret = 0;
	for (int i = 0; i < u_num_platforms; ++i)
	{
		clGetPlatformInfo(p_platforms[i], CL_PLATFORM_NAME, size, param_value, &param_value_size_ret);
		printf("Platform %p name is %s\n", p_platforms[i], param_value);
		param_value_size_ret = 0;
	}

	// 3. Получение номера CL устройства
	constexpr int32_t platform_id = 0;
	cl_device_id device_id;
	cl_uint u_num_gpu;
	clGetDeviceIDs(p_platforms[platform_id], CL_DEVICE_TYPE_DEFAULT/*CL_DEVICE_TYPE_GPU*/, 1, &device_id, &u_num_gpu);

	// 4. Получение информации о CL устройстве
	param_value_size_ret = 0;
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, size, param_value, &param_value_size_ret);
	printf("Device %p name is %s\n", device_id, param_value);


	// 5. Создание контекста
	cl_int errcode_ret;
	const cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &errcode_ret);
	if (errcode_ret != CL_SUCCESS)
	{
		printf("Error to create context");
		return 0;
	}

	// 6. Создание очереди команд
	errcode_ret = 0;
	constexpr cl_queue_properties qprop[] = { CL_QUEUE_PROPERTIES, static_cast<cl_command_queue_properties>(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE), 0 };
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, qprop, &errcode_ret);
	if (errcode_ret != CL_SUCCESS)
	{
		switch (errcode_ret)
		{
		case CL_INVALID_CONTEXT: printf("if context is not a valid context.\n");
			break;
		case CL_INVALID_DEVICE: printf("if device is not a valid device or is not associated with context.\n");
			break;
		case CL_INVALID_VALUE: printf("if values specified in properties are not valid.\n");
			break;
		case CL_INVALID_QUEUE_PROPERTIES: printf("if values specified in properties are valid but are not supported by the device.\n");
			break;
		case CL_OUT_OF_RESOURCES: printf("if there is a failure to allocate resources required by the OpenCL implementation on the device.\n");
			break;
		case CL_OUT_OF_HOST_MEMORY: printf("if there is a failure to allocate resources required by the OpenCL implementation on the host.\n");
			break;
		default:
			break;
		}
		printf("Error to create command queue");
		return 0;
	}

	// 7. Создание программы
	errcode_ret = CL_SUCCESS;
	const size_t source_size = strlen(g_pcsz_source);
	const cl_program program = clCreateProgramWithSource(context, 1, &g_pcsz_source, &source_size, &errcode_ret);
	if (errcode_ret != CL_SUCCESS)
	{
		printf("Error to create program");
		return 0;
	}

	// 8. Сборка программы
	errcode_ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
	if (errcode_ret != CL_SUCCESS)
	{
		switch (errcode_ret)
		{
		case CL_INVALID_PROGRAM: printf(" if program is not a valid program object.\n");
			break;
		case CL_INVALID_VALUE: printf(" if device_list is NULL and num_devices is greater than zero, or if device_list is not NULL and num_devices is zero.\n");
			break;
		case CL_INVALID_DEVICE: printf(" if OpenCL devices listed in device_list are not in the list of devices associated with program.\n");
			break;
		case CL_INVALID_BINARY: printf(" if program is created with clCreateWithProgramWithBinary and devices listed in device_list do not have a valid program binary loaded.\n");
			break;
		case CL_INVALID_BUILD_OPTIONS: printf(" if the build options specified by options are invalid.\n");
			break;
		case CL_INVALID_OPERATION: printf(" if the build of a program executable for any of the devices listed in device_list by a previous call to clBuildProgram for program has not completed.\n");
			break;
		case CL_COMPILER_NOT_AVAILABLE: printf(" if program is created with clCreateProgramWithSource and a compiler is not available i.e.CL_DEVICE_COMPILER_AVAILABLE specified in the table of OpenCL Device Queries for clGetDeviceInfo is set to CL_FALSE.\n");
			break;
		case CL_BUILD_PROGRAM_FAILURE: printf(" if there is a failure to build the program executable.This error will be returned if clBuildProgram does not return until the build has completed.\n");
			break;
		case CL_OUT_OF_HOST_MEMORY: printf(" if there is a failure to allocate resources required by the OpenCL implementation on the host.\n");
			break;
		default:
			break;
		}

		if (errcode_ret == CL_BUILD_PROGRAM_FAILURE) {
			size_t log_size;
			clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
			const auto log = static_cast<char*>(malloc(log_size));
			clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);
			printf("%s\n", log);
		}

		printf("Error to build program");
		return 0;
	}

	// 9. Получение ядра
	errcode_ret = CL_SUCCESS;
	cl_kernel kernel = clCreateKernel(program, "memset", &errcode_ret);
	if (errcode_ret != CL_SUCCESS)
	{
		printf("Error to create Kernel");
		return 0;
	}


	for (int k = 2; k <= 20; k++)
	{
		const auto g_cu_num_items = 2 << k;

		// 10. Создание буфера
		errcode_ret = CL_SUCCESS;
		const cl_mem buffer_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, g_cu_num_items * sizeof(cl_int), nullptr, &errcode_ret);
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Error to create buffer");
			return 0;
		}

		const auto x = new int[g_cu_num_items];
		const auto y = new int[g_cu_num_items * g_cu_num_items];
		for (int i = 0; i < g_cu_num_items; i++) {
			x[i] = 3;
			for (int j = 0; j < g_cu_num_items; j++)
			{
				y[i * g_cu_num_items + j] = 2;
			}
		}
		int a = 4;
		int sz = g_cu_num_items;
		const cl_mem buffer_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(x), x, &errcode_ret);
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Error to create buffer");
			return 0;
		}
		const cl_mem buffer_y = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * g_cu_num_items * g_cu_num_items, y, &errcode_ret);
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Error to create buffer");
			return 0;
		}		
		const cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(a), &a, &errcode_ret);
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Error to create buffer");
			return 0;
		}
		const cl_mem buffer_sz = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(sz), &sz, &errcode_ret);
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Error to create buffer");
			return 0;
		}
		// 11. Установка буфера в качестве аргумента ядра
		errcode_ret = clSetKernelArg(kernel, 0, sizeof(buffer_result), &buffer_result);
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Error to set kernel arg");
			return 0;
		}

		errcode_ret = clSetKernelArg(kernel, 1, sizeof(buffer_a), &buffer_a);
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Error to set kernel arg");
			return 0;
		}

		errcode_ret = clSetKernelArg(kernel, 2, sizeof(buffer_x), &buffer_x);
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Error to set kernel arg");
			return 0;
		}
		errcode_ret = clSetKernelArg(kernel, 3, sizeof(buffer_y), &buffer_y);
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Error to set kernel arg");
			return 0;
		}
		errcode_ret = clSetKernelArg(kernel, 4, sizeof(buffer_sz), &buffer_sz);
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Error to set kernel arg");
			return 0;
		}
		const auto gpu_start_time = high_resolution_clock::now();
		// 12. Запуск ядра
		errcode_ret = CL_SUCCESS;
		const size_t u_global_work_size = g_cu_num_items * g_cu_num_items;
		errcode_ret = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &u_global_work_size, nullptr, 0, nullptr, nullptr);
		clFinish(queue);
		if (errcode_ret != CL_SUCCESS)
		{
			switch (errcode_ret)
			{
			case CL_INVALID_PROGRAM_EXECUTABLE: printf("  if there is no successfully built program executable available for device associated with command_queue..\n");
				break;
			case CL_INVALID_COMMAND_QUEUE: printf("  if command_queue is not a valid command - queue..\n");
				break;
			case CL_INVALID_KERNEL: printf("  if kernel is not a valid kernel object..\n");
				break;
			case  CL_INVALID_CONTEXT: printf("  if context associated with command_queue and kernel is not the same or if the context associated with command_queue and events in event_wait_list are not the same..\n");
				break;
			case CL_INVALID_KERNEL_ARGS: printf("  if the kernel argument values have not been specified..\n");
				break;
			case CL_INVALID_WORK_DIMENSION: printf(" if work_dim is not a valid value(i.e.a value between 1 and 3)..\n");
				break;
			case CL_INVALID_WORK_GROUP_SIZE: printf(" if local_work_size is specified and number of work - items specified by global_work_size is not evenly divisable by size of work - group given by local_work_size or does not match the work - group size specified for kernel using the __attribute__((reqd_work_group_size(X, Y, Z))) qualifier in program source..\n");
				break;
			case CL_INVALID_WORK_ITEM_SIZE: printf(" if the number of work - items specified in any of local_work_size[0], ... local_work_size[work_dim - 1] is greater than the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES[0], ....CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1]..\n");
				break;
			case CL_INVALID_GLOBAL_OFFSET: printf(" if global_work_offset is not NULL..\n");
				break;
			case CL_OUT_OF_RESOURCES: printf(" if there is a failure to queue the execution instance of kernel on the command - queue because of insufficient resources needed to execute the kernel.For example, the explicitly specified local_work_size causes a failure to execute the kernel because of insufficient resources such as registers or local memory.Another example would be the number of read - only image args used in kernel exceed the CL_DEVICE_MAX_READ_IMAGE_ARGS value for device or the number of write - only image args used in kernel exceed the CL_DEVICE_MAX_WRITE_IMAGE_ARGS value for device or the number of samplers used in kernel exceed CL_DEVICE_MAX_SAMPLERS for device..\n");
				break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:  printf(" if there is a failure to allocate memory for data store associated with image or buffer objects specified as arguments to kernel..\n");
				break;
			case CL_INVALID_EVENT_WAIT_LIST: printf(" if event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL and num_events_in_wait_list is 0, or if event objects in event_wait_list are not valid events..\n");
				break;
			case CL_OUT_OF_HOST_MEMORY: printf(" if there is a failure to allocate resources required by the OpenCL implementation on the host..\n");
				break;
			default:
				break;
			}

			printf("Error to create context");
			return 0;
		}
		const auto gpu_end_time = high_resolution_clock::now();

		// 13. Отображение буфера в память управляющего узла
		errcode_ret = CL_SUCCESS;
		auto* result = static_cast<cl_int*>(clEnqueueMapBuffer(queue, buffer_result, CL_TRUE, CL_MAP_READ, 0, g_cu_num_items * sizeof(cl_int), 0, nullptr, nullptr, &errcode_ret));
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Error to create context");
			return 0;
		}

		// 14. Использование результатов
		/*errcode_ret = CL_SUCCESS;
		for (int i = 0; i < g_cu_num_items; ++i)
			std::cout << i << " = " << result[i] << "; ";
		std::cout << std::endl;
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Error to create context");
			return 0;
		}*/
		printf("gpu (sz = %i) = %lld\n", g_cu_num_items, duration_cast<microseconds>(gpu_end_time - gpu_start_time).count());

		// 15. Завершение отображения буфера
		errcode_ret = CL_SUCCESS;
		clEnqueueUnmapMemObject(queue, buffer_result, result, 0, nullptr, nullptr);
		clReleaseMemObject(buffer_result);


		// cpu
		const auto cpu_res = new int[g_cu_num_items];
		const auto cpu_start_time = high_resolution_clock::now();
		for (int i = 0; i < g_cu_num_items; i++)
		{
			cpu_res[i] = 0;
			for(int j = 0; j < g_cu_num_items; j++)
			{
				cpu_res[i] = cpu_res[i] + a * y[i * sz + j] * x[j];
			}
		}
		const auto cpu_end_time = high_resolution_clock::now();
		printf("CPU (sz = %i) = %lld\n", g_cu_num_items, duration_cast<microseconds>(cpu_end_time - cpu_start_time).count());
	}

	// 16. Удаление объектов и освобождение памяти управляющего узла
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	delete[] p_platforms;
	return  0;
}

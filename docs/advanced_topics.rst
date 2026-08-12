.. _advanced_topics:

Advanced Topics
===============

This section covers advanced usage patterns, configuration options, and integration strategies for developers looking to leverage the full capabilities of the HEonGPU library.

Configuration (``defines.h``)
-----------------------------

The ``src/include/heongpu/kernel/defines.h`` file is an essential configuration header that contains key settings defining the library's limits and default behaviors. Advanced users can modify these values to tune the library for specific hardware or demanding use cases.

Key settings include:

* **Polynomial Degree**: ``MAX_POLY_DEGREE`` (default: 65536) and ``MIN_POLY_DEGREE`` (default: 4096) define the supported range for polynomial degrees.
* **Modulus Bit-Length**: These constants specify the valid bit-lengths for user-defined and general modulus values, ensuring cryptographic security.
* **Galois Key Capability**: ``MAX_SHIFT`` (default: 8) controls the maximum rotation capability for default Galois key generation. If your application requires more rotation steps, this value must be increased.
* **Memory Pool Sizes**: The initial and maximum sizes for the device (GPU) and host (CPU) memory pools are defined as percentages of available system memory. By default, the GPU pool is initialized to 50% of VRAM and can grow to 80%, while the pinned host memory pool is initialized to 10% of RAM and can grow to 20%. These values can be adjusted for systems with different memory capacities or for applications with particularly large memory footprints.

Multiparty Computation (MPC)
----------------------------

HEonGPU includes support for secure **Multi-Party Computation (MPC)** protocols, enabling multiple parties to collaboratively compute on encrypted data. This is achieved through Multiparty Homomorphic Encryption (MHE) capabilities, providing a framework for threshold encryption models such as `N-out-of-N`.

A key feature of the MPC implementation is **Collective (Distributed) Bootstrapping** for both the BFV and CKKS schemes. This powerful technique, based on designs by Mouchet et al. and Balle et al., allows a group of parties to jointly refresh a ciphertext, resetting its noise level without a trusted dealer. The implementation merges the share creation and re-encryption steps into a single, efficient CUDA kernel launch. This allows deep multi-party workloads to continue running entirely on the GPU without pausing to reset noise, which is critical for complex, collaborative privacy-preserving applications.

Using HEonGPU in a Downstream CMake Project
-------------------------------------------

To integrate HEonGPU into your own CMake project, first ensure the library is installed on your system. The installed library provides the necessary config files to make integration seamless. In your project's ``CMakeLists.txt``, you can use ``find_package`` to locate and link against the library.

.. code-block:: cmake

    # Set the project language to include CUDA
    project(<your-project> LANGUAGES CXX CUDA)

    # Find the CUDA Toolkit, which is a dependency
    find_package(CUDAToolkit REQUIRED)
    
    # ... your other project configurations ...

    # Find the HEonGPU package
    find_package(HEonGPU REQUIRED)

    # ... define your executable target ...
    add_executable(<your-target> main.cpp)

    # Link your application against the HEonGPU library and the CUDA runtime
    target_link_libraries(<your-target> PRIVATE HEonGPU::heongpu CUDA::cudart)

    # Enable separable compilation for CUDA, which is often required
    set_target_properties(<your-target> PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

Against a HEonGPU built with ``USE_HIP=ON``, the same project uses the HIP language and the HIP runtime instead. Sources that include the HEonGPU headers must be compiled as HIP, because the headers pull in rocThrust.

.. code-block:: cmake

    # Set the project language to include HIP
    project(<your-project> LANGUAGES CXX HIP)

    # Find HIP, which is a dependency
    find_package(hip REQUIRED)

    # ... your other project configurations ...

    # Find the HEonGPU package
    find_package(HEonGPU REQUIRED)

    # ... define your executable target ...
    add_executable(<your-target> main.cpp)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

    # Link your application against the HEonGPU library and the HIP runtime
    target_link_libraries(<your-target> PRIVATE HEonGPU::heongpu hip::host)

The AMD build of the library holds relocatable device code, so the device link happens when your own target is linked, and only the HIP compiler driver in HIP link mode can perform it. Your project must enable the HIP language for this to be possible at all, as the ``project(<your-project> LANGUAGES CXX HIP)`` line above does; ``enable_language(HIP)`` does the same, and ``find_package(hip)`` does not. Given that, the installed package supplies the driver on its own: it records HIP as the link interface language of ``HEonGPU::heongpu`` and carries ``-fgpu-rdc`` on its interface, so every CMake target that links it, directly or through another library of your own, is driven by the HIP compiler and gets that flag without asking. What CMake supplies only for a target that compiles a source of its own as HIP is HIP link mode itself; the snippet above has it because ``main.cpp`` is compiled as HIP. A target that links the library but has no HIP source of its own is therefore driven by the HIP compiler but not in HIP link mode, and fails with undefined references to ``__hip_fatbin_*`` and ``__hip_gpubin_handle_*``; ask for the mode explicitly there:

.. code-block:: cmake

    # Only needed when the target itself has no HIP sources
    target_link_options(<your-target> PRIVATE --hip-link)

If your project does not enable HIP, the installed package does not enable the language for you, and nothing set on the target itself makes up for it. CMake links such a target with your C++ compiler, which rejects the interface flag before anything else is tried (``unrecognized command-line option '-fgpu-rdc'``); adding ``--hip-link`` there only earns a second rejected option, and ``LINKER_LANGUAGE HIP`` is not available, because without the language CMake has no HIP link rule and stops at generate time with ``Missing variable is: CMAKE_HIP_LINK_EXECUTABLE``. Enabling the HIP language does not oblige you to compile any of your own sources as HIP; it only gives CMake the compiler it needs to finish the device link.

Naming the installed archive by path instead of going through ``find_package`` keeps that requirement and adds to it, since nothing then supplies the driver or the flag: the link is driven by your C++ compiler and fails with the same undefined references, and reaching a working link takes ``LINKER_LANGUAGE HIP`` on the target together with ``--hip-link`` and ``-fgpu-rdc`` on its link line, in a project where HIP is enabled. Outside CMake, compile and link with ``hipcc -fgpu-rdc``.

Project Roadmap
---------------

HEonGPU is an actively developing project with a clear vision for the future. The roadmap indicates a strategic expansion from accelerating core cryptographic primitives to enabling complex, end-to-end privacy-preserving systems.

Planned enhancements include:

* **Support for Additional Schemes**: The library plans to add support for the **BGV** scheme to expand its cryptographic capabilities.
* **Python Wrapper**: To make the library more accessible to the data science and machine learning communities, a Python wrapper is on the roadmap.
* **Multi-GPU Support**: The architecture is planned to be extended to support multi-GPU configurations, including a multi-GPU memory pool structure, to facilitate the execution of even larger-scale applications.

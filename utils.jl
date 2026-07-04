# Utility functions for khape simulations

import CUDA  # for version-robust qualified access (CUDA.free_memory etc.)
using CUDA: devices, device!, functional, totalmem, name

#+++ Grid sizing functions
function closest_factor_number(primes::NTuple{3, Int}, target::Int)
    closest_number = 1
    min_difference = abs(target - closest_number)
    # We will iterate over different combinations of powers of primes
    for i in 0:15  # You can adjust this loop depth
        for j in 0:15
            for k in 0:15
                # Generate the product of primes with different powers
                product = primes[1]^i * primes[2]^j * primes[3]^k
                diff = abs(target - product)
                if diff < min_difference
                    min_difference = diff
                    closest_number = product
                end
            end
        end
    end
    return closest_number
end

function closest_factor_number(primes::NTuple{2, Int}, target::Int)
    closest_number = 1
    min_difference = abs(target - closest_number)
    # We will iterate over different combinations of powers of primes
    for i in 0:15 # You can adjust this loop depth
        for j in 0:15
            # Generate the product of primes with different powers
            product = primes[1]^i * primes[2]^j
            diff = abs(target - product)
            if diff < min_difference
                min_difference = diff
                closest_number = product
            end
        end
    end
    return closest_number
end

function closest_factor_number(primes::NTuple{1, Int}, target::Int)
    closest_number = 1
    min_difference = abs(target - closest_number)
    # We will iterate over different combinations of powers of primes
    for i in 0:15 # You can adjust this loop depth
        # Generate the product of primes with different powers
        product = primes[1]^i
        diff = abs(target - product)
        if diff < min_difference
            min_difference = diff
            closest_number = product
        end
    end
    return closest_number
end
#---

#+++ GPU Status Functions
function get_gpu_memory_usage(gpu_device)
    total_mem = totalmem(gpu_device) |> Float64
    free_mem  = isdefined(CUDA, :free_memory) ? CUDA.free_memory() : CUDA.available_memory()  # CUDA.jl v6 renamed available_memory → free_memory
    used_mem  = total_mem - free_mem
    return total_mem, free_mem, used_mem
end

function show_gpu_status()
    # Check if CUDA is available
    if !functional()
        return
    end

    # Get number of available GPUs
    num_devices = length(devices())

    println("="^70)
    println("GPU Status Report")
    println("="^70)
    println("Number of GPUs available: $num_devices")
    println()

    # Iterate through all available GPUs
    for (i, gpu_device) in enumerate(devices())
        # Set current device
        device!(gpu_device)

        # Get device information
        gpu_name  = name(gpu_device)
        total_mem, free_mem, used_mem = get_gpu_memory_usage(gpu_device)

        # Convert to GB for readability
        used_gb = used_mem / (1024^3)
        total_gb = total_mem / (1024^3)
        usage_percent = (used_mem / total_mem) * 100

        # Display information
        println("GPU $i: $gpu_name")
        println("  Used Memory:  $(round(used_gb, digits=2)) GB")
        println("  Total Memory: $(round(total_gb, digits=2)) GB")
        println("  Usage:        $(round(usage_percent, digits=1))%")

        # Add a visual progress bar
        bar_length = 30
        filled_length = Int(round(usage_percent / 100 * bar_length))
        bar = "█" ^ filled_length * "░" ^ (bar_length - filled_length)
        println("  [$(bar)] $(round(usage_percent, digits=1))%")
        println()
        println("Double check with CUDA's native function:")
        if isdefined(CUDA, :memory_status)
            CUDA.memory_status()    # CUDA.jl ≤ 5 native pool report
        elseif isdefined(CUDA, :pool_status)
            CUDA.pool_status()      # CUDA.jl v6 renamed memory_status → pool_status
        else                        # last-resort numeric fallback
            free_gb = CUDA.free_memory() / 1024^3
            println("  Device memory: $(round(free_gb, digits=2)) GiB free / $(round(total_gb, digits=2)) GiB total")
        end
    end

    println("=" ^ 70)
end
#---

use std::{fmt::Display, mem::MaybeUninit, hint::black_box};

use cudarc::driver::{
    sys::{
        cuDeviceGetAttribute, cuDeviceGetCount, cuDeviceGetName, cuDeviceGetProperties,
        cuDeviceTotalMem_v2, CUdevice_attribute,
    },
    CudaDeviceBuilder,
};

struct PrintWithDots<T>(T);
impl<T: ToString> Display for PrintWithDots<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.0.to_string();
        let segments = (s.len() + 2) / 3 - 1;
        let base = s.len() % 3;
        let mut result = String::with_capacity(segments * 4 + base);
        if base > 0 {
            result.push_str(&s[..base]);
            if segments > 0 {
                result.push('.');
            }
        }
        for i in 0..segments {
            result.push_str(&s[base + 3 * i..base + 3 * i + 3]);
            if i < segments - 1 {
                result.push('.');
            }
        }
        f.pad(&result)
    }
}

unsafe fn _main() {
    // cuda context is required, blackbox to NOT optimize it away
    let _cuda = black_box(CudaDeviceBuilder::new(0).build().unwrap());
    let mut device_count = MaybeUninit::uninit();
    cuDeviceGetCount(device_count.as_mut_ptr())
        .result()
        .unwrap();
    for device_id in 0..device_count.assume_init() {
        let mut properties = MaybeUninit::uninit();
        cuDeviceGetProperties(properties.as_mut_ptr(), device_id)
            .result()
            .unwrap();
        let properties = properties.assume_init();
        let mut name = [0u8; 128];
        cuDeviceGetName(name.as_mut_ptr().cast(), name.len() as i32, device_id)
            .result()
            .unwrap();
        let mut bytes = MaybeUninit::uninit();
        cuDeviceTotalMem_v2(bytes.as_mut_ptr(), device_id)
            .result()
            .unwrap();
        println!(
            "{}: {}",
            device_id,
            String::from_utf8_lossy(&name[..name.iter().position(|n| n == &0).unwrap_or(0)])
        );
        println!(
            "  Total memory:       {:>15} bytes",
            PrintWithDots(bytes.assume_init().to_string())
        );
        println!(
            "  Shared memory:      {:>15} bytes",
            PrintWithDots(properties.sharedMemPerBlock)
        );
        println!(
            "  Constant memory:    {:>15} bytes",
            PrintWithDots(properties.totalConstantMemory)
        );
        println!(
            "  Block registers:    {:>15}",
            PrintWithDots(properties.regsPerBlock)
        );
        let mut warp_size = MaybeUninit::uninit();
        cuDeviceGetAttribute(
            warp_size.as_mut_ptr(),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE,
            device_id,
        )
        .result()
        .unwrap();
        println!(
            "  Warp size:          {:>15}",
            PrintWithDots(warp_size.assume_init())
        );
        println!(
            "  Max grid size:      {:>15}",
            PrintWithDots(properties.maxGridSize.into_iter().max().unwrap())
        );
        println!(
            "  Max block size:     {:>15}",
            PrintWithDots(properties.maxThreadsDim.into_iter().max().unwrap())
        );
        println!(
            "  Threads per block:  {:>15}",
            PrintWithDots(properties.maxThreadsPerBlock)
        );
        println!(
            "  SIMD width:         {:>15}",
            PrintWithDots(properties.SIMDWidth)
        );
        println!();
    }
}

fn main() {
    unsafe {
        _main();
    }
}

use crate::allocator::CachingAllocator;
use rand::Rng;
use rand_distr::StandardNormal;

pub type TensorId = usize;

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    strides[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn shape_size(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

// =========================================================================
// CPU tensor
// =========================================================================

#[cfg(feature = "cpu")]
pub struct GpuTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub size: usize,
    pub requires_grad: bool,
    pub grad: Option<TensorId>,
    pub adam_m: Option<Vec<f32>>,
    pub adam_v: Option<Vec<f32>>,
}

#[cfg(feature = "cpu")]
impl GpuTensor {
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected = 1usize;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected {
                return false;
            }
            expected *= self.shape[i];
        }
        true
    }
}

// =========================================================================
// CUDA tensor
// =========================================================================

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
pub struct GpuTensor {
    pub data: CudaSlice<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub size: usize,
    pub requires_grad: bool,
    pub grad: Option<TensorId>,
    pub adam_m: Option<CudaSlice<f32>>,
    pub adam_v: Option<CudaSlice<f32>>,
}

#[cfg(feature = "cuda")]
impl GpuTensor {
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected = 1usize;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected {
                return false;
            }
            expected *= self.shape[i];
        }
        true
    }
}

// =========================================================================
// TensorStore
// =========================================================================

pub struct TensorStore {
    pub(crate) tensors: Vec<Option<GpuTensor>>,
    pub(crate) free_ids: Vec<TensorId>,
    alloc: CachingAllocator,
}

// Shared logic (works for both CPU and CUDA)
impl TensorStore {
    fn insert(&mut self, t: GpuTensor) -> TensorId {
        if let Some(id) = self.free_ids.pop() {
            self.tensors[id] = Some(t);
            id
        } else {
            let id = self.tensors.len();
            self.tensors.push(Some(t));
            id
        }
    }

    pub fn get(&self, id: TensorId) -> &GpuTensor {
        self.tensors[id].as_ref().expect("tensor already freed")
    }

    pub fn get_mut(&mut self, id: TensorId) -> &mut GpuTensor {
        self.tensors[id].as_mut().expect("tensor already freed")
    }

    pub fn shape(&self, id: TensorId) -> &[usize] {
        &self.get(id).shape
    }

    pub fn size(&self, id: TensorId) -> usize {
        self.get(id).size
    }

    pub fn set_requires_grad(&mut self, id: TensorId, requires: bool) {
        self.get_mut(id).requires_grad = requires;
    }

    pub fn get_grad(&self, id: TensorId) -> Option<TensorId> {
        self.get(id).grad
    }

    pub fn ones_like(&mut self, id: TensorId) -> TensorId {
        let shape = self.get(id).shape.clone();
        self.ones(&shape)
    }

    pub fn ensure_grad(&mut self, id: TensorId) -> TensorId {
        if let Some(g) = self.get(id).grad {
            return g;
        }
        let shape = self.get(id).shape.clone();
        let grad_id = self.zeros(&shape);
        self.get_mut(id).grad = Some(grad_id);
        grad_id
    }
}

// =========================================================================
// CPU-specific TensorStore methods
// =========================================================================

#[cfg(feature = "cpu")]
impl TensorStore {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            free_ids: Vec::new(),
            alloc: CachingAllocator::new(),
        }
    }

    pub fn free(&mut self, id: TensorId) {
        if let Some(t) = self.tensors[id].take() {
            self.alloc.dealloc(t.data);
            self.free_ids.push(id);
        }
    }

    pub fn data(&self, id: TensorId) -> &[f32] {
        &self.get(id).data
    }

    pub fn data_mut(&mut self, id: TensorId) -> &mut [f32] {
        &mut self.get_mut(id).data
    }

    pub fn to_host(&self, id: TensorId) -> Vec<f32> {
        let t = self.get(id);
        if t.is_contiguous() {
            t.data.clone()
        } else {
            self.make_contiguous_data(id)
        }
    }

    pub fn get_scalar(&self, id: TensorId) -> f32 {
        self.get(id).data[0]
    }

    pub fn zero_grad(&mut self, id: TensorId) {
        if let Some(grad_id) = self.get(id).grad {
            let size = self.get(grad_id).size;
            let data = self.data_mut(grad_id);
            for i in 0..size {
                data[i] = 0.0;
            }
        }
    }

    pub fn zeros(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let data = self.alloc.alloc(size);
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
        })
    }

    pub fn ones(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut data = self.alloc.alloc(size);
        data.iter_mut().for_each(|x| *x = 1.0);
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
        })
    }

    pub fn rand(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut data = self.alloc.alloc(size);
        let mut rng = rand::thread_rng();
        data.iter_mut().for_each(|x| *x = rng.gen::<f32>());
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
        })
    }

    pub fn randn(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut data = self.alloc.alloc(size);
        let mut rng = rand::thread_rng();
        data.iter_mut().for_each(|x| *x = rng.sample::<f32, _>(StandardNormal));
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
        })
    }

    pub fn from_slice(&mut self, src: &[f32], shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut data = self.alloc.alloc(size);
        data[..size].copy_from_slice(&src[..size]);
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
        })
    }

    pub fn from_vec(&mut self, src: Vec<f32>, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data: src, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
        })
    }

    pub fn accumulate_grad(&mut self, dst: TensorId, src: TensorId) {
        let grad_id = self.ensure_grad(dst);
        let size = self.get(src).size;
        let src_data = self.get(src).data.clone();
        let grad_data = self.data_mut(grad_id);
        for i in 0..size {
            grad_data[i] += src_data[i];
        }
    }

    /// In-place addition: dst[i] += src[i]
    pub fn add_inplace(&mut self, dst: TensorId, src: TensorId) {
        let size = self.get(src).size;
        let src_data = self.get(src).data.clone();
        let dst_data = self.data_mut(dst);
        for i in 0..size {
            dst_data[i] += src_data[i];
        }
    }

    fn make_contiguous_data(&self, id: TensorId) -> Vec<f32> {
        let t = self.get(id);
        let size = t.size;
        let ndim = t.shape.len();
        let mut out = vec![0.0f32; size];
        let out_strides = compute_strides(&t.shape);
        for i in 0..size {
            let mut src_idx = 0;
            let mut rem = i;
            for d in 0..ndim {
                let coord = rem / out_strides[d];
                rem %= out_strides[d];
                src_idx += coord * t.strides[d];
            }
            out[i] = t.data[src_idx];
        }
        out
    }

    pub fn ensure_contiguous(&mut self, id: TensorId) -> TensorId {
        if self.get(id).is_contiguous() {
            return id;
        }
        let data = self.make_contiguous_data(id);
        let shape = self.get(id).shape.clone();
        self.from_vec(data, &shape)
    }
}

// =========================================================================
// CUDA-specific TensorStore methods
// =========================================================================

#[cfg(feature = "cuda")]
use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig};

#[cfg(feature = "cuda")]
fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(feature = "cuda")]
impl TensorStore {
    pub fn new() -> Self {
        // Force CUDA device initialization
        let _ = crate::device::GpuDevice::instance();
        Self {
            tensors: Vec::new(),
            free_ids: Vec::new(),
            alloc: CachingAllocator::new(),
        }
    }

    pub fn free(&mut self, id: TensorId) {
        if let Some(t) = self.tensors[id].take() {
            self.alloc.dealloc(t.data);
            self.free_ids.push(id);
        }
    }

    pub fn dev_ptr(&self, id: TensorId) -> u64 {
        *self.get(id).data.device_ptr()
    }

    pub fn to_host(&self, id: TensorId) -> Vec<f32> {
        let dev = crate::device::GpuDevice::instance();
        let t = self.get(id);
        if t.is_contiguous() {
            dev.stream.clone_dtoh(&t.data).unwrap()
        } else {
            let host = dev.stream.clone_dtoh(&t.data).unwrap();
            let size = t.size;
            let ndim = t.shape.len();
            let out_strides = compute_strides(&t.shape);
            let mut out = vec![0.0f32; size];
            for i in 0..size {
                let mut src_idx = 0;
                let mut rem = i;
                for d in 0..ndim {
                    let coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    src_idx += coord * t.strides[d];
                }
                out[i] = host[src_idx];
            }
            out
        }
    }

    pub fn get_scalar(&self, id: TensorId) -> f32 {
        let dev = crate::device::GpuDevice::instance();
        let v = dev.stream.clone_dtoh(&self.get(id).data).unwrap();
        v[0]
    }

    pub fn zero_grad(&mut self, id: TensorId) {
        if let Some(grad_id) = self.get(id).grad {
            let size = self.get(grad_id).size;
            let dev = crate::device::GpuDevice::instance();
            let out_ptr = *self.get(grad_id).data.device_ptr();
            let func = dev.get_func("fill_f32");
            unsafe {
                dev.stream.launch_builder(func)
                    .arg(&out_ptr)
                    .arg(&0.0f32)
                    .arg(&(size as i32))
                    .launch(launch_cfg(size as u32))
                    .unwrap();
            }
        }
    }

    pub fn zeros(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let data = self.alloc.alloc(size);
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
        })
    }

    pub fn ones(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let data = self.alloc.alloc(size);
        let strides = compute_strides(shape);
        let id = self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
        });
        let dev = crate::device::GpuDevice::instance();
        let ptr = *self.get(id).data.device_ptr();
        let func = dev.get_func("fill_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&ptr)
                .arg(&1.0f32)
                .arg(&(size as i32))
                .launch(launch_cfg(size as u32))
                .unwrap();
        }
        id
    }

    pub fn rand(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut host = vec![0.0f32; size];
        let mut rng = rand::thread_rng();
        host.iter_mut().for_each(|x| *x = rng.gen::<f32>());
        self.from_slice(&host, shape)
    }

    pub fn randn(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut host = vec![0.0f32; size];
        let mut rng = rand::thread_rng();
        host.iter_mut().for_each(|x| *x = rng.sample::<f32, _>(StandardNormal));
        self.from_slice(&host, shape)
    }

    pub fn from_slice(&mut self, src: &[f32], shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let dev = crate::device::GpuDevice::instance();
        let data = dev.stream.clone_htod(&src[..size]).unwrap();
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
        })
    }

    pub fn from_vec(&mut self, src: Vec<f32>, shape: &[usize]) -> TensorId {
        self.from_slice(&src, shape)
    }

    /// GPU-side gradient accumulation: grad[i] += src[i]
    pub fn accumulate_grad(&mut self, dst: TensorId, src: TensorId) {
        let grad_id = self.ensure_grad(dst);
        let size = self.get(src).size;
        let dev = crate::device::GpuDevice::instance();
        let src_ptr = *self.get(src).data.device_ptr();
        let grad_ptr = *self.get(grad_id).data.device_ptr();
        let func = dev.get_func("add_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&grad_ptr)
                .arg(&grad_ptr)
                .arg(&src_ptr)
                .arg(&(size as i32))
                .launch(launch_cfg(size as u32))
                .unwrap();
        }
    }

    /// In-place addition on GPU: dst[i] += src[i]
    pub fn add_inplace(&mut self, dst: TensorId, src: TensorId) {
        let size = self.get(src).size;
        let dev = crate::device::GpuDevice::instance();
        let src_ptr = *self.get(src).data.device_ptr();
        let dst_ptr = *self.get(dst).data.device_ptr();
        let func = dev.get_func("add_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&dst_ptr)
                .arg(&dst_ptr)
                .arg(&src_ptr)
                .arg(&(size as i32))
                .launch(launch_cfg(size as u32))
                .unwrap();
        }
    }

    pub fn ensure_contiguous(&mut self, id: TensorId) -> TensorId {
        if self.get(id).is_contiguous() {
            return id;
        }
        let data = self.to_host(id);
        let shape = self.get(id).shape.clone();
        self.from_vec(data, &shape)
    }
}

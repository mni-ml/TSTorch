use crate::tensor::{TensorId, TensorStore};

#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use cudarc::driver::{DevicePtr, LaunchConfig};

#[cfg(feature = "cuda")]
fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

// =========================================================================
// AdamW step
// =========================================================================

#[cfg(feature = "cpu")]
pub fn adamw_step(
    param_ids: &[TensorId],
    lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32,
    step: u32, store: &mut TensorStore,
) {
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);

    for &pid in param_ids {
        let grad_id = match store.get(pid).grad {
            Some(g) => g,
            None => continue,
        };

        let size = store.size(pid);
        let grad_data = store.to_host(grad_id);

        if store.get(pid).adam_m.is_none() {
            store.get_mut(pid).adam_m = Some(vec![0.0f32; size]);
            store.get_mut(pid).adam_v = Some(vec![0.0f32; size]);
        }

        let param = store.get_mut(pid);
        let m = param.adam_m.as_mut().unwrap();
        let v = param.adam_v.as_mut().unwrap();
        let data = &mut param.data;

        for i in 0..size {
            let g = grad_data[i];

            if weight_decay > 0.0 {
                data[i] *= 1.0 - lr * weight_decay;
            }

            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;

            data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

#[cfg(feature = "cuda")]
pub fn adamw_step(
    param_ids: &[TensorId],
    lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32,
    step: u32, store: &mut TensorStore,
) {
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    let dev = GpuDevice::instance();

    for &pid in param_ids {
        let grad_id = match store.get(pid).grad {
            Some(g) => g,
            None => continue,
        };

        let size = store.size(pid);

        if store.get(pid).adam_m.is_none() {
            let m = dev.stream.alloc_zeros(size).unwrap();
            let v = dev.stream.alloc_zeros(size).unwrap();
            store.get_mut(pid).adam_m = Some(m);
            store.get_mut(pid).adam_v = Some(v);
        }

        let tensors_ptr = store.tensors.as_mut_ptr();
        unsafe {
            let param_tensor = (*tensors_ptr.add(pid)).as_mut().unwrap();
            let grad_tensor = (*tensors_ptr.add(grad_id)).as_ref().unwrap();

            let param_ptr = *param_tensor.data.device_ptr();
            let m_ptr = *param_tensor.adam_m.as_ref().unwrap().device_ptr();
            let v_ptr = *param_tensor.adam_v.as_ref().unwrap().device_ptr();
            let grad_ptr = *grad_tensor.data.device_ptr();

            let func = dev.get_func("adamw_step_f32");
            dev.stream.launch_builder(func)
                .arg(&param_ptr)
                .arg(&m_ptr)
                .arg(&v_ptr)
                .arg(&grad_ptr)
                .arg(&lr)
                .arg(&beta1)
                .arg(&beta2)
                .arg(&eps)
                .arg(&weight_decay)
                .arg(&bc1)
                .arg(&bc2)
                .arg(&(size as i32))
                .launch(launch_cfg(size as u32))
                .unwrap();
        }
    }
}

// =========================================================================
// Gradient norm
// =========================================================================

#[cfg(feature = "cpu")]
pub fn grad_norm(param_ids: &[TensorId], store: &TensorStore) -> f32 {
    let mut norm_sq = 0.0f64;
    for &pid in param_ids {
        if let Some(grad_id) = store.get(pid).grad {
            let data = store.data(grad_id);
            for &v in data {
                norm_sq += (v as f64) * (v as f64);
            }
        }
    }
    (norm_sq as f32).sqrt()
}

#[cfg(feature = "cuda")]
pub fn grad_norm(param_ids: &[TensorId], store: &TensorStore) -> f32 {
    let mut norm_sq = 0.0f64;
    for &pid in param_ids {
        if let Some(grad_id) = store.get(pid).grad {
            let data = store.to_host(grad_id);
            for &v in &data {
                norm_sq += (v as f64) * (v as f64);
            }
        }
    }
    (norm_sq as f32).sqrt()
}

// =========================================================================
// Gradient clipping
// =========================================================================

#[cfg(feature = "cpu")]
pub fn clip_grad_norm(param_ids: &[TensorId], max_norm: f32, store: &mut TensorStore) {
    let norm = grad_norm(param_ids, store);
    if norm > max_norm {
        let scale = max_norm / norm;
        for &pid in param_ids {
            if let Some(grad_id) = store.get(pid).grad {
                let data = store.data_mut(grad_id);
                for v in data.iter_mut() {
                    *v *= scale;
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub fn clip_grad_norm(param_ids: &[TensorId], max_norm: f32, store: &mut TensorStore) {
    let norm = grad_norm(param_ids, store);
    if norm > max_norm {
        let scale = max_norm / norm;
        let dev = GpuDevice::instance();
        for &pid in param_ids {
            if let Some(grad_id) = store.get(pid).grad {
                let size = store.size(grad_id);
                let grad_ptr = store.dev_ptr(grad_id);
                let func = dev.get_func("grad_clip_f32");
                unsafe {
                    dev.stream.launch_builder(func)
                        .arg(&grad_ptr)
                        .arg(&scale)
                        .arg(&(size as i32))
                        .launch(launch_cfg(size as u32))
                        .unwrap();
                }
            }
        }
    }
}

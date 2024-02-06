use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum KinodeMlLibrary {
    PyTorch,
    TensorFlow,
    Keras,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum KinodeMlDataType {
    Float16,
    BFloat16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Uint8,
    Uint16,
    Uint32,
    //...
}

#[derive(Debug, Serialize, Deserialize)]
pub struct KinodeMlRequest {
    pub library: KinodeMlLibrary,
    pub data_shape: Vec<u64>,
    pub data_type: KinodeMlDataType,
    pub model_bytes: Vec<u8>,
    pub data_bytes: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct KinodeMlResponse {
    pub library: KinodeMlLibrary,
    pub data_shape: Vec<u64>,
    pub data_type: KinodeMlDataType,
    pub data_bytes: Vec<u8>,
}

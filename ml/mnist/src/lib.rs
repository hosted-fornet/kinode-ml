use kinode_process_lib::{call_init, get_blob, println, Address, LazyLoadBlob, Request};

mod ml_types;
use ml_types::{KinodeMlLibrary, KinodeMlDataType, KinodeMlRequest, KinodeMlResponse};

wit_bindgen::generate!({
    path: "wit",
    world: "process",
    exports: {
        world: Component,
    },
});

const MODEL: &[u8] = include_bytes!("./TFKeras.h5");
const DATA: &[u8] = include_bytes!("./test3.png");

call_init!(init);

fn init(_our: Address) {
    println!("kinode_mnist: begin");

    let input = image::load_from_memory_with_format(DATA, image::ImageFormat::Png)
        .unwrap()
        .to_luma8()
        .into_raw()
        .iter()
        .map(|&p| !p)
        .map(|p| p as f32 / 255.0)
        .flat_map(|p| p.to_ne_bytes().to_vec())
        .collect();

    let _response = Request::new()
        .target("our@ml:ml:sys".parse::<Address>().unwrap())
        .body(serde_json::to_vec(&serde_json::json!("Run")).unwrap())
        .blob(LazyLoadBlob {
            mime: None,
            bytes: rmp_serde::to_vec_named(&KinodeMlRequest {
                library: KinodeMlLibrary::Keras,
                data_shape: vec![1, 784],
                data_type: KinodeMlDataType::Float32,
                model_bytes: MODEL.to_vec(),
                data_bytes: input,
            }).unwrap(),
        })
        .send_and_await_response(15)
        .unwrap();

    let Some(LazyLoadBlob { ref bytes, .. }) = get_blob() else {
        panic!("a");
    };
    let KinodeMlResponse { library, data_shape, data_type, data_bytes } = rmp_serde::from_slice(bytes).unwrap() else {
        panic!("b");
    };
    let output: Vec<f32> = data_bytes.chunks(4)
        .map(|chunk| {
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(arr)
        })
        .collect();
    let prediction = output.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap();
    // println!(
    //     "library: {:?}\n\rdata_shape: {:?}\n\rdata_type: {:?}\n\rdata_bytes: {:?}\n\routput: {:?}\r\nthe given number was: {}",
    //     library,
    //     data_shape,
    //     data_type,
    //     data_bytes,
    //     output,
    //     prediction,
    // );
    println!("output: {:?}\n\rthe given number was: {}", output, prediction);
}

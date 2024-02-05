use kinode_process_lib::http;
use kinode_process_lib::kernel_types::MessageType;
use kinode_process_lib::{
    await_message, call_init, get_blob, println, Address, LazyLoadBlob, Message, Request, Response,
};

mod ml_types;

wit_bindgen::generate!({
    path: "wit",
    world: "process",
    exports: {
        world: Component,
    },
});

struct Connection {
    channel_id: u32,
}

fn is_expected_channel_id(
    connection: &Option<Connection>,
    channel_id: &u32,
) -> anyhow::Result<bool> {
    let Some(Connection { channel_id: ref current_channel_id }) = connection else {
        return Err(anyhow::anyhow!("foo"));
    };

    Ok(channel_id == current_channel_id)
}

fn handle_ws_message(
    connection: &mut Option<Connection>,
    message: Message,
) -> anyhow::Result<()> {
    match serde_json::from_slice::<http::HttpServerRequest>(message.body())? {
        http::HttpServerRequest::Http(_) => {
            // TODO: response?
            return Err(anyhow::anyhow!("foo"));
        }
        http::HttpServerRequest::WebSocketOpen { channel_id, .. } => {
            *connection = Some(Connection { channel_id });
        }
        http::HttpServerRequest::WebSocketClose(ref channel_id) => {
            if !is_expected_channel_id(connection, channel_id)? {
                // TODO: response?
                return Err(anyhow::anyhow!("foo"));
            }
            *connection = None;
        }
        http::HttpServerRequest::WebSocketPush { ref channel_id, ref message_type } => {
            if !is_expected_channel_id(connection, channel_id)? {
                // TODO: response?
                return Err(anyhow::anyhow!("foo"));
            }
            match message_type {
                http::WsMessageType::Binary => {
                    Response::new()
                        .body(serde_json::to_vec(&serde_json::json!("Run"))?)
                        .inherit(true)
                        .send()?;
                }
                http::WsMessageType::Ping => {
                    let _ = Request::new()
                        .target("our@http_server:distro:sys".parse::<Address>()?)
                        .body(serde_json::to_vec(&http::HttpServerAction::WebSocketPush {
                            channel_id: *channel_id,
                            message_type: http::WsMessageType::Pong,
                        })?)
                        .inherit(true)
                        .send_and_await_response(5)?;
                }
                _ => {
                    // TODO: response; handle other types?
                    return Err(anyhow::anyhow!("foo"));
                }
            }
        }
    }
    Ok(())
}

fn handle_message(
    connection: &mut Option<Connection>,
) -> anyhow::Result<()> {
    let Ok(message) = await_message() else {
        return Ok(());
    };

    if serde_json::json!("Run") != serde_json::from_slice::<serde_json::Value>(message.body())? {
        handle_ws_message(connection, message)?;
    } else {
        let Some(Connection { channel_id }) = connection else {
            panic!("");
        };

        Request::new()
            .target("our@http_server:distro:sys".parse::<Address>()?)
            .body(serde_json::to_vec(&http::HttpServerAction::WebSocketExtPushOutgoing {
                channel_id: *channel_id,
                message_type: http::WsMessageType::Binary,
                desired_reply_type: MessageType::Response,
            })?)
            .expects_response(60)
            //.expects_response(15)
            .inherit(true)
            .send()?;
    }

    Ok(())
}

call_init!(init);
fn init(our: Address) {
    println!("{our}: begin");

    let mut connection: Option<Connection> = None;

    http::bind_ext_path("/").unwrap();

    loop {
        match handle_message(&mut connection) {
            Ok(()) => {}
            Err(e) => {
                println!("{our}: error: {:?}", e);
            }
        };
    }
}

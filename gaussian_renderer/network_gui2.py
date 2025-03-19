import time

import viser
import viser.transforms as tf

import numpy as np

host = "0.0.0.0"
port = 8080

server = {
    "server": None,
    "render_type": None,
}


class dummygui:
    def __init__(self, value):
        self.value = value


def init(initial_value='rendered color'):
    global host, port, server
    server["server"] = viser.ViserServer(host=host, port=port)
    server["server"].scene.world_axes.visible = True
    server["server"].scene.set_up_direction(direction = '+z')
    server["render_type"] = dummygui("debug")

    @server["server"].on_client_connect
    def _(client: viser.ClientHandle) -> None:

        # This will run whenever we get a new camera!
        #@client.camera.on_update
        #def _(_: viser.CameraHandle) -> None:
        #    print(f"New camera on client {client.client_id}!")

        # Show the client ID in the GUI.
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

        global server
        server["render_type"] = client.gui.add_dropdown(
            "Render Type",
            options=[
                "rendered color",
                "base color",
                "refl strength",
                "normal",
                "envmap cood1",
                "envmap cood2",
            ],
            initial_value=initial_value
        )

        gui_reset_up = client.gui.add_button(
            "Reset up direction",
            hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )

        @gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

    return server


def try_connect():
    if server["server"] is None:
        return

    while True:
        # Get all currently connected clients.
        clients = server["server"].get_clients()
        print("Connected client IDs", clients.keys())

        if len(clients) == 0:
            time.sleep(0.5)
        else:
            server["client"] = clients[0]
            break


def on_gui_change():
    if server["server"] is None:
        return ""

    return server["render_type"].value
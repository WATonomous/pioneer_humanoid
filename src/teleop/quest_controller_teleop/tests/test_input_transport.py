"""Pose transport, reference changes, loss detection and controller button mapping."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2]/"src/pioneer_humanoid")]
from transport import validate_packet


def raw_sample(sequence=0, stamp=0., angle=0., tracked=True, epoch=0):
    q = [np.sin(angle/2), 0., 0., np.cos(angle/2)]
    return {"version": 1, "head": [0., 0., 0., 1.],
            "left": {"tracked": True, "position": [0., 1., 0.], "orientation": q,
                     "trigger": 0., "clutch": True} if tracked else None,
            "right": None, "sample": {"sequence": sequence, "time_ms": stamp,
            "epoch": epoch}}


class InputTransportTests(unittest.TestCase):
    def test_wire_order_and_quaternion_sign_are_correct(self):
        raw = raw_sample(angle=np.pi/2)
        rotation = validate_packet(raw)["left"].rotation
        np.testing.assert_allclose(rotation @ [0., 0., 1.], [0., -1., 0.], atol=1e-12)
        raw["left"]["orientation"] = [-v for v in raw["left"]["orientation"]]
        np.testing.assert_allclose(validate_packet(raw)["left"].rotation, rotation)

    def test_invalid_ordering_metadata_is_rejected(self):
        for key, value in (("sequence", True), ("time_ms", float("nan")), ("epoch", -1)):
            raw = raw_sample()
            raw["sample"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_packet(raw)


@unittest.skipUnless(importlib.util.find_spec("aiohttp"), "Optional aiohttp transport test dependency unavailable")
class WebSocketRoundTripTests(unittest.IsolatedAsyncioTestCase):
    async def test_pose_recenter_tracking_loss_and_y_reset_round_trip(self):
        import asyncio
        import contextlib
        from types import SimpleNamespace
        from unittest.mock import patch
        from aiohttp import ClientSession, WSServerHandshakeError, web
        from transport import Shared, serve
        with tempfile.TemporaryDirectory() as d:
            shared = Shared()
            ready = asyncio.get_running_loop().create_future()
            base = web.TCPSite
            class Site(base):
                async def start(self):
                    await super().start()
                    ready.set_result(self._server.sockets[0].getsockname()[1])
            with patch.object(web, "TCPSite", Site):
                task = asyncio.create_task(serve(shared, SimpleNamespace(host="127.0.0.1", port=0, cert=None)))
                try:
                    port = await asyncio.wait_for(ready, timeout=5)
                    async with ClientSession() as desktop:   # an editor's port preview must not take control
                        with self.assertRaises(WSServerHandshakeError) as refused:
                            await desktop.ws_connect(f"http://127.0.0.1:{port}/ws")
                        self.assertEqual(refused.exception.status, 403)
                    async with ClientSession(headers={"User-Agent": "Mozilla/5.0 OculusBrowser/38.0"}) as client:
                        async with client.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                            raw = raw_sample(1, 10., angle=.3)
                            await ws.send_json(raw)
                            feedback = await ws.receive_json(timeout=2)
                            self.assertNotIn("input", feedback)
                            self.assertNotIn("tracking", feedback)
                            np.testing.assert_allclose(shared.packet["left"].rotation, validate_packet(raw)["left"].rotation)
                            shared.interrupted = {"left": False, "right": False}
                            await ws.send_json(raw_sample(2, 20., angle=.3, epoch=1))
                            await ws.receive_json(timeout=2)
                            self.assertTrue(shared.interrupted["left"])
                            shared.interrupted = {"left": False, "right": False}
                            await ws.send_json(raw_sample(3, 30., tracked=False, epoch=1))
                            await ws.receive_json(timeout=2)
                            await ws.send_json(raw_sample(4, 40., epoch=1))
                            await ws.receive_json(timeout=2)
                            self.assertTrue(shared.interrupted["left"])
                            # Y is armed only after a released tracked frame.
                            reset = raw_sample(5, 50., epoch=1)
                            reset["left"]["reset_scene"] = True
                            await ws.send_json(reset)
                            await ws.receive_json(timeout=2)
                            self.assertEqual(shared.reset_sequence, 1)
                            reset["sample"].update(sequence=6, time_ms=60.)
                            await ws.send_json(reset)
                            await ws.receive_json(timeout=2)
                            self.assertEqual(shared.reset_sequence, 1)
                            # An out-of-order pose is ignored; the next valid
                            # frame still reaches the latest-value mailbox.
                            await ws.send_json(raw_sample(5, 50., epoch=1))
                            await ws.send_json(raw_sample(7, 70., epoch=1))
                            await ws.receive_json(timeout=2)
                            self.assertEqual(shared.packet["sample"]["sequence"], 7)
                        await asyncio.sleep(.1)   # the server releases the closed owner
                        # A tab open but not in VR must not lock out the tab in VR (several tabs on the same page
                        # once left the headset stuck on "reconnecting"); while that tab streams, others are ignored.
                        url = f"http://127.0.0.1:{port}/ws"
                        async with client.ws_connect(url) as idle, client.ws_connect(url) as worn:
                            await worn.send_json(raw_sample(1, 10., angle=.5))
                            await worn.receive_json(timeout=2)
                            worn_rotation = validate_packet(raw_sample(angle=.5))["left"].rotation
                            np.testing.assert_allclose(shared.packet["left"].rotation, worn_rotation)
                            await idle.send_json(raw_sample(1, 10., angle=.1))
                            feedback = await idle.receive_json(timeout=2)
                            self.assertEqual(feedback["status"], "Another headset tab has control")
                            np.testing.assert_allclose(shared.packet["left"].rotation, worn_rotation)
                finally:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task


@unittest.skipUnless(importlib.util.find_spec("quickjs"), "Optional QuickJS browser test runtime unavailable")
class BrowserPacketTests(unittest.TestCase):
    def setUp(self):
        import quickjs
        self.js = quickjs.Context()
        self.js.eval('''
            let clock=100, sent=[];
            const performance={now:()=>clock};
            const elements=new Map();
            function element(id){if(!elements.has(id))elements.set(id,{textContent:'',getContext:()=>({})});return elements.get(id);}
            const document={querySelector:element,getElementById:element,createElement:element,addEventListener:()=>{}};
            const navigator={};
            const location={protocol:'http:',host:'localhost:8444'};
            const window={addEventListener:()=>{}};
            const setTimeout=()=>1, clearTimeout=()=>{};
            class WebSocket{static OPEN=1;constructor(){this.readyState=1;this.bufferedAmount=0;}send(x){sent.push(JSON.parse(x));}}
        ''')
        script = (HERE/"index.html").read_text().split("<script>", 1)[1].split("</script>", 1)[0]
        self.js.eval(script)

    def test_each_90hz_frame_can_send_and_backpressure_skips_without_queueing(self):
        self.js.eval('''session={requestAnimationFrame:()=>{}};
            for(let i=0;i<90;i++){clock=100+i*1000/90;draw(clock,{getViewerPose:()=>null});}''')
        packets = json.loads(self.js.eval("JSON.stringify(sent)"))
        self.assertEqual(len(packets), 90)
        self.assertEqual(packets[-1]["sample"]["sequence"], 90)
        self.js.eval("ws.bufferedAmount=100;draw(clock,{getViewerPose:()=>null});")
        self.assertEqual(self.js.eval("sent.length"), 90)
        self.js.eval("ws.bufferedAmount=0;draw(clock,{getViewerPose:()=>null});")
        self.assertEqual(self.js.eval("sent[90].sample.sequence"), 91)

    def test_grip_space_xyzw_mapping_and_recenter_stop(self):
        self.js.eval('''reference={};feedback={scene:'isaac'};
            const pose={emulatedPosition:false,transform:{position:{x:1,y:2,z:3},orientation:{x:.1,y:.2,z:.3,w:.9}}};
            const source={gripSpace:{},handedness:'left',gamepad:{mapping:'xr-standard',buttons:[{value:.4},{pressed:false},{},{pressed:true},{pressed:true},{pressed:true}]}};
            const output=controller({getPose:()=>pose},source);sendPacket({version:1,head:[0,0,0,1],left:output,right:null});''')
        packet = json.loads(self.js.eval("JSON.stringify(sent[0])"))
        self.assertEqual(packet["left"]["orientation"], [.1, .2, .3, .9])
        self.assertEqual(packet["left"]["position"], [1, 2, 3])
        self.assertEqual(packet["left"]["trigger"], .4)
        self.assertFalse(packet["left"]["clutch"])
        self.assertTrue(packet["left"]["precision"])
        self.assertTrue(packet["left"]["calibrate"])
        self.assertTrue(packet["left"]["reset_scene"])
        self.js.eval("clock=120;stopped();")
        stop = json.loads(self.js.eval("JSON.stringify(sent[1])"))
        self.assertEqual(stop["sample"]["epoch"], 1)
        self.assertIsNone(stop["left"])

    def test_browser_reports_why_a_controller_pose_is_rejected(self):
        self.js.eval("reference={};const source={gripSpace:{},gamepad:{mapping:'xr-standard'}};")
        reason = self.js.eval("controller({getPose:()=>null},source).reason")
        self.assertEqual(reason, "pose_unavailable")
        reason = self.js.eval("controller({getPose:()=>({emulatedPosition:true})},source).reason")
        self.assertEqual(reason, "emulated_position")
        self.assertFalse(self.js.eval("controller({getPose:()=>({emulatedPosition:true})},source).tracked"))


if __name__ == "__main__":
    unittest.main()

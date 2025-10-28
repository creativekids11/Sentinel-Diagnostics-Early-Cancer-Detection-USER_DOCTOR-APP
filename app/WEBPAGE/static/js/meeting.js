"use strict";

const callBtn = document.getElementById("call");
const localVideo = document.getElementById("local-video");
const remoteVideo = document.getElementById("remote-video");

let pc;
let localStream;
localStream = await navigator.mediaDevices.getUserMedia({
	audio: true,
	video: true,
});
localVideo.srcObject = localStream;

const signaling = new BroadcastChannel("webrtc");
signaling.onmessage = async (event) => {
	if (!localStream) {
		console.log("not ready yet");
		return;
	}
	switch (event.data.type) {
		case "offer":
			if (pc) {
				console.error("existing peerconnection");
				return;
			}
			await PeerConnection();
			await pc.setRemoteDescription(event.data);

			const answer = await pc.createAnswer();
			signaling.postMessage({ type: "answer", sdp: answer.sdp });
			await pc.setLocalDescription(answer);
			break;
		case "answer":
			if (!pc) {
				console.error("no peerconnection");
				return;
			}
			await pc.setRemoteDescription(event.data);
			break;
		case "candidate":
			if (!pc) {
				return;
			}
			if (!event.data.candidate) {
				await pc.addIceCandidate(null);
			} else {
				await pc.addIceCandidate(event.data);
			}
			break;
		case "ready":
			// A second tab joined. This tab will initiate a call unless in a call already.
			if (pc) {
				console.log("Already in call, ignoring");
				return;
			}
			await PeerConnection();

			const offer = await pc.createOffer();
			signaling.postMessage({ type: "offer", sdp: offer.sdp });
			await pc.setLocalDescription(offer);
			break;
		case "bye":
			if (pc) {
				pc.close();
				pc = null;
			}
			remoteVideo.srcObject = null;
			localStream = null;
			break;
		default:
			console.log("unhandled", event);
			break;
	}
};

callBtn.onclick = async () => {
	if (callBtn.dataset.state === "off") {
		callBtn.dataset.state = "on";
		callBtn.innerText = "Leave";

		signaling.postMessage({ type: "ready" });
	} else if (callBtn.dataset.state === "on") {
		callBtn.dataset.state = "off";
		callBtn.innerText = "Call";
		signaling.postMessage({ type: "bye" });
	}
};

async function PeerConnection() {
	const jsonFile = await fetch("/static/Javascript/index.json");
	const jsonFileData = await jsonFile.json();

	// WebRTC Configuration
	let config = {
		iceServers: [
			{
				urls: "stun:stun.l.google.com:19302",
			},
			{
				urls: "turn:global.relay.metered.ca:443",
				username: jsonFileData.username,
				credential: jsonFileData.credential,
			},
			{
				urls: "turns:global.relay.metered.ca:443?transport=tcp",
				username: jsonFileData.username,
				credential: jsonFileData.credential,
			},
		],
	};
	pc = new RTCPeerConnection(config);
	pc.onicecandidate = (e) => {
		const message = {
			type: "candidate",
			candidate: null,
		};
		if (e.candidate) {
			message.candidate = e.candidate.candidate;
			message.sdpMid = e.candidate.sdpMid;
			message.sdpMLineIndex = e.candidate.sdpMLineIndex;
		}
		signaling.postMessage(message);
	};
	pc.ontrack = (e) => (remoteVideo.srcObject = e.streams[0]);
	localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));
}

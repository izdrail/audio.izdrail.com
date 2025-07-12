const recordBtn = document.getElementById("recordBtn");
const transcriptionEl = document.getElementById("transcription");
const responseEl = document.getElementById("response");
const audioPlayer = document.getElementById("audioPlayer");

let mediaRecorder;
let audioChunks = [];

recordBtn.addEventListener("click", async () => {
    // ✅ Check browser support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("🚫 Your browser doesn't support microphone access. Try Chrome or Firefox.");
        return;
    }

    if (mediaRecorder?.state === "recording") {
        mediaRecorder.stop();
        recordBtn.textContent = "Start Recording";
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const audioFile = new File([audioBlob], "recording.wav");

            // 1. Transcribe
            const formData = new FormData();
            formData.append("audio", audioFile);

            const transcriptionRes = await fetch("/transcribe", {
                method: "POST",
                body: formData
            });

            const transcriptionData = await transcriptionRes.json();
            const transcript = transcriptionData.text;
            transcriptionEl.textContent = transcript;

            // 2. LLM Response (simulate)
            const llmResponse = `You said: "${transcript}". 你说的是：「${transcript}」. Ты сказал: "${transcript}"`;
            responseEl.textContent = llmResponse;

            // 3. TTS
            const ttsRes = await fetch("/tts", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: llmResponse })
            });

            if (ttsRes.ok) {
                const ttsBlob = await ttsRes.blob();
                const audioURL = URL.createObjectURL(ttsBlob);
                audioPlayer.src = audioURL;
                audioPlayer.play();
            } else {
                console.error("TTS failed:", await ttsRes.text());
                alert("🧨 TTS generation failed.");
            }
        };

        mediaRecorder.start();
        recordBtn.textContent = "Stop Recording";
    } catch (err) {
        console.error("Mic error:", err);
        alert("🚫 Mic access denied or unavailable.");
    }
});

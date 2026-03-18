import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import ragaData from './ragas.json'

const API_URL = 'https://raga-identifier-production.up.railway.app'

export default function App() {
  const [state, setState] = useState('idle')
  const [predictions, setPredictions] = useState(null)
  const [error, setError] = useState(null)
  const [waveform, setWaveform] = useState(Array.from({ length: 80 }, () => 8 + Math.random() * 30))
  const [recordingTime, setRecordingTime] = useState(0)

  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const fileInputRef = useRef(null)
  const animFrameRef = useRef(null)
  const analyserRef = useRef(null)

  const startWaveformAnimation = (analyser) => {
    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)
    const bars = 80
    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw)
      analyser.getByteTimeDomainData(dataArray)
      const step = Math.floor(bufferLength / bars)
      const newWave = Array.from({ length: bars }, (_, i) => {
        const val = dataArray[i * step] / 128.0
        return Math.max(4, Math.abs(val - 1) * 80)
      })
      setWaveform(newWave)
    }
    draw()
  }

  const stopWaveformAnimation = () => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current)
      animFrameRef.current = null
    }
    setWaveform(Array.from({ length: 80 }, () => 8 + Math.random() * 30))
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const audioCtx = new AudioContext()
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 1024
      source.connect(analyser)
      analyserRef.current = analyser
      startWaveformAnimation(analyser)

      const recorder = new MediaRecorder(stream)
      chunksRef.current = []
      recorder.ondataavailable = e => chunksRef.current.push(e.data)
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/wav' })
        stream.getTracks().forEach(t => t.stop())
        audioCtx.close()
        stopWaveformAnimation()
        submitAudio(blob, 'recording.wav')
      }
      recorder.start()
      mediaRecorderRef.current = recorder
      setState('recording')
      setRecordingTime(0)
      timerRef.current = setInterval(() => setRecordingTime(t => t + 1), 1000)
    } catch {
      setError('Microphone access denied.')
    }
  }

  const stopRecording = () => {
    clearInterval(timerRef.current)
    mediaRecorderRef.current?.stop()
    setState('processing')
  }

  const handleFileUpload = e => {
    const file = e.target.files[0]
    if (!file) return
    setState('processing')
    submitAudio(file, file.name)
  }

  const submitAudio = async (blob, filename) => {
    setError(null)
    const form = new FormData()
    form.append('file', blob, filename)
    try {
      const res = await axios.post(`${API_URL}/predict`, form)
      setPredictions(res.data)
      setState('result')
    } catch (e) {
      setError(e.response?.data?.detail || 'Something went wrong.')
      setState('idle')
    }
  }

  const reset = () => {
    setState('idle')
    setPredictions(null)
    setError(null)
    setRecordingTime(0)
    setWaveform(Array.from({ length: 80 }, () => 8 + Math.random() * 30))
  }

  const topRaga = predictions?.top_raga
  const ragaInfo = topRaga ? ragaData[topRaga] : null

  return (
    <div style={styles.app}>
      <div style={styles.header}>
        <div style={styles.headerTop}>
          <span style={styles.title}>Raga Identifier</span>
          <span style={styles.dot} />
        </div>
        <div style={styles.subtitle}>Carnatic music recognition</div>
      </div>

      {(state === 'idle' || state === 'recording') && (
        <>
          <div style={styles.inputGrid}>
            <div
              style={{ ...styles.card, ...(state === 'recording' ? styles.cardActive : {}), cursor: 'pointer' }}
              onClick={state === 'idle' ? startRecording : stopRecording}
            >
              <div style={{ ...styles.recordBtn, ...(state === 'recording' ? styles.recordBtnActive : {}) }}>
                <div style={{ ...styles.recordDot, ...(state === 'recording' ? styles.recordDotActive : {}) }} />
              </div>
              <div style={styles.inputLabel}>{state === 'recording' ? `Stop  ${recordingTime}s` : 'Record'}</div>
              <div style={styles.inputDesc}>{state === 'recording' ? 'Tap to stop and identify' : 'Sing or play into your mic'}</div>
            </div>

            <div style={{ ...styles.card, cursor: 'pointer' }} onClick={() => fileInputRef.current?.click()}>
              <div style={styles.uploadBox}>↑</div>
              <div style={styles.inputLabel}>Upload</div>
              <div style={styles.inputDesc}>Drop a .wav or .mp3 file</div>
              <input ref={fileInputRef} type="file" accept=".wav,.mp3,.m4a" style={{ display: 'none' }} onChange={handleFileUpload} />
            </div>
          </div>

          <div style={styles.waveCard}>
            <div style={styles.waveLabel}>{state === 'recording' ? 'Live input' : 'Waveform'}</div>
            <div style={styles.waveBars}>
              {waveform.map((h, i) => (
                <div key={i} style={{
                  ...styles.waveBar,
                  height: h,
                  opacity: state === 'recording' ? 0.5 + (h / 80) * 0.5 : 0.4 + (h / 38) * 0.5,
                  background: state === 'recording' ? '#c4826a' : '#dbb99e',
                  transition: state === 'recording' ? 'height 0.05s ease' : 'none'
                }} />
              ))}
            </div>
          </div>
        </>
      )}

      {state === 'processing' && (
        <div style={{ ...styles.card, textAlign: 'center', padding: '48px 24px' }}>
          <div style={styles.processingDots}>
            {[0, 1, 2].map(i => (
              <div key={i} style={{ ...styles.processingDot, animationDelay: `${i * 0.2}s` }} />
            ))}
          </div>
          <div style={styles.processingText}>Identifying raga...</div>
        </div>
      )}

      {state === 'result' && predictions && (
        <>
          <div style={styles.divider} />
          <div style={styles.card}>
            <div style={styles.eyebrow}>Identified raga</div>
            <div style={styles.ragaName}>{topRaga}</div>
            <div style={styles.confidenceTag}>{predictions.confidence}% confident</div>
            {ragaInfo && (
              <div style={styles.detailsGrid}>
                <div style={styles.detailPill}>
                  <div style={styles.detailKey}>Arohanam</div>
                  <div style={styles.detailVal}>{ragaInfo.arohanam}</div>
                </div>
                <div style={styles.detailPill}>
                  <div style={styles.detailKey}>Avarohanam</div>
                  <div style={styles.detailVal}>{ragaInfo.avarohanam}</div>
                </div>
              </div>
            )}
          </div>

          <div style={styles.card}>
            <div style={styles.cardTitle}>Top predictions</div>
            {predictions.predictions.map((p, i) => (
              <div key={i} style={styles.confRow}>
                <div style={styles.confName}>{p.raga}</div>
                <div style={styles.confTrack}>
                  <div style={{ ...styles.confFill, width: `${p.confidence}%`, opacity: i === 0 ? 1 : 0.5 - i * 0.05 }} />
                </div>
                <div style={styles.confPct}>{p.confidence}%</div>
              </div>
            ))}
          </div>

          {ragaInfo?.similar && (
            <>
              <div style={styles.cardTitle}>Similar ragas</div>
              <div style={styles.similarGrid}>
                {ragaInfo.similar.map((r, i) => (
                  <div key={i} style={styles.similarCard}>
                    <div style={styles.similarName}>{r}</div>
                    <div style={styles.similarTag}>{ragaData[r] ? `${ragaData[r].arohanam.split(' ').length} swaras` : 'Related raga'}</div>
                  </div>
                ))}
              </div>
            </>
          )}

          <button style={styles.resetBtn} onClick={reset}>Try another</button>
        </>
      )}

      {error && <div style={styles.error}>{error}</div>}
    </div>
  )
}

const styles = {
  app: { maxWidth: 760, margin: '0 auto', padding: '40px 24px 60px', fontFamily: "'DM Sans', sans-serif", background: '#f5f2ee', minHeight: '100vh', color: '#2c2c2c' },
  header: { marginBottom: 36 },
  headerTop: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 },
  title: { fontFamily: 'Georgia, serif', fontSize: 28, fontWeight: 500, color: '#1e1e1e', letterSpacing: -0.3 },
  dot: { width: 7, height: 7, borderRadius: '50%', background: '#c4826a', display: 'inline-block' },
  subtitle: { fontSize: 13, color: '#9a9082', fontWeight: 300, letterSpacing: 0.3 },
  inputGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 },
  card: { background: '#fff', border: '1px solid #e8e2da', borderRadius: 12, padding: '28px 20px', textAlign: 'center', marginBottom: 12 },
  cardActive: { borderColor: '#c4826a', background: '#fdf8f6' },
  recordBtn: { width: 64, height: 64, borderRadius: '50%', background: '#fdf0eb', border: '1.5px solid #c4826a', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 14px' },
  recordBtnActive: { background: '#fce8e0' },
  recordDot: { width: 20, height: 20, borderRadius: '50%', background: '#c4826a' },
  recordDotActive: { borderRadius: 4 },
  uploadBox: { width: 64, height: 64, borderRadius: 10, background: '#f5f2ee', border: '1.5px dashed #c8c0b4', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 14px', fontSize: 20, color: '#b0a898' },
  inputLabel: { fontSize: 15, fontWeight: 500, color: '#2c2c2c', marginBottom: 4 },
  inputDesc: { fontSize: 12, color: '#9a9082', lineHeight: 1.5 },
  waveCard: { background: '#fff', border: '1px solid #e8e2da', borderRadius: 12, padding: '16px 20px', marginBottom: 28 },
  waveLabel: { fontSize: 11, color: '#b0a898', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 10 },
  waveBars: { display: 'flex', alignItems: 'center', gap: 2, height: 60 },
  waveBar: { flex: 1, background: '#dbb99e', borderRadius: 2 },
  processingDots: { display: 'flex', gap: 8, justifyContent: 'center', marginBottom: 16 },
  processingDot: { width: 8, height: 8, borderRadius: '50%', background: '#c4826a', animation: 'bounce 0.8s ease-in-out infinite' },
  processingText: { fontSize: 14, color: '#9a9082' },
  divider: { height: 1, background: '#e0dbd4', margin: '0 0 20px' },
  eyebrow: { fontSize: 11, letterSpacing: 1.5, textTransform: 'uppercase', color: '#b0a898', marginBottom: 8 },
  ragaName: { fontFamily: 'Georgia, serif', fontSize: 42, fontWeight: 500, color: '#1e1e1e', letterSpacing: -0.5, lineHeight: 1.1, marginBottom: 8 },
  confidenceTag: { display: 'inline-block', background: '#eef6ee', color: '#5a8a5a', fontSize: 12, fontWeight: 500, padding: '3px 10px', borderRadius: 20, marginBottom: 20 },
  detailsGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 },
  detailPill: { background: '#faf8f5', border: '1px solid #ede8e0', borderRadius: 8, padding: '10px 14px' },
  detailKey: { fontSize: 10, letterSpacing: 1, textTransform: 'uppercase', color: '#b0a898', marginBottom: 3 },
  detailVal: { fontSize: 14, color: '#3c3530', fontFamily: 'Georgia, serif', fontStyle: 'italic' },
  cardTitle: { fontSize: 11, letterSpacing: 1.5, textTransform: 'uppercase', color: '#b0a898', marginBottom: 16 },
  confRow: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 },
  confName: { width: 130, fontSize: 14, color: '#3c3530', fontFamily: 'Georgia, serif', flexShrink: 0 },
  confTrack: { flex: 1, height: 5, background: '#f0ebe4', borderRadius: 3, overflow: 'hidden' },
  confFill: { height: '100%', background: '#c4826a', borderRadius: 3, transition: 'width 0.6s ease' },
  confPct: { width: 36, textAlign: 'right', fontSize: 12, color: '#b0a898' },
  similarGrid: { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 20 },
  similarCard: { background: '#fff', border: '1px solid #e8e2da', borderRadius: 10, padding: '14px 16px' },
  similarName: { fontSize: 14, fontFamily: 'Georgia, serif', color: '#2c2c2c', marginBottom: 3 },
  similarTag: { fontSize: 11, color: '#b0a898' },
  resetBtn: { width: '100%', padding: '12px', borderRadius: 10, background: '#fff', border: '1px solid #e8e2da', fontSize: 14, color: '#9a9082', cursor: 'pointer', marginTop: 8 },
  error: { background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 10, padding: '12px 16px', fontSize: 13, color: '#b91c1c', marginTop: 12 },
}

import { useState, useRef } from 'react'
import axios from 'axios'
import ragaData from './ragas.json'

const API_URL = 'https://smashgod23-raga-identifier-api.hf.space'

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
const SA_OPTIONS = (() => {
  const opts = [{ label: 'Auto-detect', value: '' }]
  // Chromatic from C2 to B4 covers the realistic Sa range for male and female vocalists.
  for (let m = 36; m <= 71; m++) {
    const hz = 440 * Math.pow(2, (m - 69) / 12)
    const name = NOTE_NAMES[m % 12] + (Math.floor(m / 12) - 1)
    opts.push({ label: `${name}  ·  ${hz.toFixed(1)} Hz`, value: hz.toFixed(2) })
  }
  return opts
})()

export default function App() {
  const [state, setState] = useState('idle')
  const [predictions, setPredictions] = useState(null)
  const [error, setError] = useState(null)
  const [waveform, setWaveform] = useState(Array.from({ length: 80 }, () => 8 + Math.random() * 30))
  const [recordingTime, setRecordingTime] = useState(0)
  const [feedback, setFeedback] = useState(null)
  const [selectedRaga, setSelectedRaga] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [showDropdown, setShowDropdown] = useState(false)
  const [audioId, setAudioId] = useState('')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [tonicHz, setTonicHz] = useState('')
  const [processingMsg, setProcessingMsg] = useState('')
  const processingTimerRef = useRef(null)

  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const fileInputRef = useRef(null)
  const animFrameRef = useRef(null)

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
      startWaveformAnimation(analyser)
      // Use the browser's actual recording format (webm on Chrome, ogg on Firefox, mp4 on Safari)
      const mimeTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg', 'audio/mp4']
      const mimeType = mimeTypes.find(t => MediaRecorder.isTypeSupported(t)) || ''
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      chunksRef.current = []
      recorder.ondataavailable = e => chunksRef.current.push(e.data)
      recorder.onstop = () => {
        const actualMime = recorder.mimeType
        const ext = actualMime.includes('ogg') ? '.ogg' : actualMime.includes('mp4') ? '.m4a' : '.webm'
        const blob = new Blob(chunksRef.current, { type: actualMime })
        stream.getTracks().forEach(t => t.stop())
        audioCtx.close()
        stopWaveformAnimation()
        submitAudio(blob, `recording${ext}`)
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

  const startProcessing = (steps) => {
    setProcessingMsg(steps[0])
    let i = 1
    clearInterval(processingTimerRef.current)
    processingTimerRef.current = setInterval(() => {
      if (i < steps.length) {
        setProcessingMsg(steps[i])
        i++
      }
    }, 3000)
  }

  const stopProcessing = () => {
    clearInterval(processingTimerRef.current)
    processingTimerRef.current = null
  }

  const audioSteps = [
    'Analyzing audio...',
    'Detecting pitch contour...',
    'Estimating tonic (Sa)...',
    'Extracting swara distribution...',
    'Matching against 40 ragas...'
  ]

  const youtubeSteps = [
    'Fetching audio from YouTube...',
    'This may take a moment for longer videos...',
    'Analyzing audio...',
    'Detecting pitch contour...',
    'Estimating tonic (Sa)...',
    'Extracting swara distribution...',
    'Matching against 40 ragas...'
  ]

  const stopRecording = () => {
    clearInterval(timerRef.current)
    mediaRecorderRef.current?.stop()
    setState('processing')
    startProcessing(audioSteps)
  }

  const handleFileUpload = e => {
    const file = e.target.files[0]
    if (!file) return
    setState('processing')
    startProcessing(audioSteps)
    submitAudio(file, file.name)
  }

  const submitAudio = async (blob, filename) => {
    setError(null)
    const form = new FormData()
    form.append('file', blob, filename)
    if (tonicHz) form.append('tonic_hz', tonicHz)
    try {
      const res = await axios.post(`${API_URL}/predict`, form)
      setPredictions(res.data)
      setAudioId(res.data.audio_id || '')
      setState('result')
      setFeedback(null)
    } catch (e) {
      setError(e.response?.data?.detail || 'Something went wrong.')
      setState('idle')
    } finally {
      stopProcessing()
    }
  }

  const submitYoutubeUrl = async () => {
    if (!youtubeUrl.trim()) return
    setState('processing')
    setError(null)
    startProcessing(youtubeSteps)
    try {
      const ytBody = { url: youtubeUrl }
      if (tonicHz) ytBody.tonic_hz = parseFloat(tonicHz)
      const res = await axios.post(`${API_URL}/predict-youtube`, ytBody)
      setPredictions(res.data)
      setAudioId(res.data.audio_id || '')
      setState('result')
      setFeedback(null)
    } catch (e) {
      setError(e.response?.data?.detail || 'Could not process YouTube link.')
      setState('idle')
    } finally {
      stopProcessing()
    }
  }

  const reset = () => {
    setState('idle')
    setPredictions(null)
    setError(null)
    setRecordingTime(0)
    setFeedback(null)
    setSelectedRaga('')
    setSearchQuery('')
    setAudioId('')
    setYoutubeUrl('')
    setWaveform(Array.from({ length: 80 }, () => 8 + Math.random() * 30))
  }

  const submitFeedback = (actualRaga, wasCorrect) => {
    setFeedback('submitted')
    axios.post(`${API_URL}/feedback`, {
      predicted_raga: predictions.top_raga,
      actual_raga: actualRaga,
      was_correct: wasCorrect,
      confidence: predictions.confidence,
      audio_filename: audioId
    }).catch(() => {})
  }

  const topRaga = predictions?.top_raga
  const ragaInfo = topRaga ? ragaData[topRaga] : null
  const filteredRagas = Object.keys(ragaData).filter(r =>
    r.toLowerCase().includes(searchQuery.toLowerCase())
  )

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
          <div style={styles.tonicPicker}>
            <div>
              <div style={styles.tonicLabel}>My Sa <span style={styles.tonicHint}>optional</span></div>
              <div style={styles.tonicDesc}>Set this for solo voice — auto-detect works best with a tambura drone</div>
            </div>
            <select
              style={styles.tonicSelect}
              value={tonicHz}
              onChange={e => setTonicHz(e.target.value)}
              disabled={state === 'recording'}
            >
              {SA_OPTIONS.map(opt => (
                <option key={opt.label} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
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
          <div style={styles.youtubeCard}>
            <div style={styles.youtubeHeader}>
              <div style={styles.youtubeIcon}>▶</div>
              <div>
                <div style={styles.inputLabel}>YouTube Link</div>
                <div style={styles.inputDesc}>Paste a link to identify the raga</div>
              </div>
            </div>
            <div style={styles.youtubeInputRow}>
              <input
                style={styles.youtubeInput}
                placeholder="https://youtube.com/watch?v=..."
                value={youtubeUrl}
                onChange={e => setYoutubeUrl(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && submitYoutubeUrl()}
              />
              <button
                style={{ ...styles.youtubeBtn, opacity: youtubeUrl.trim() ? 1 : 0.5 }}
                onClick={submitYoutubeUrl}
                disabled={!youtubeUrl.trim()}
              >
                Identify
              </button>
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
          <div style={styles.processingText}>{processingMsg}</div>
        </div>
      )}

      {state === 'result' && predictions && (
        <>
          <div style={styles.divider} />
          <div style={styles.card}>
            <div style={styles.eyebrow}>Identified raga</div>
            <div style={styles.ragaName}>{topRaga}</div>
            <div style={styles.tagRow}>
              <div style={styles.confidenceTag}>{predictions.confidence}% confident</div>
              {predictions.tonic_note && (
                <div style={predictions.tonic_overridden ? styles.saTagSet : styles.saTag}>
                  {predictions.tonic_overridden ? 'Your Sa' : 'Detected Sa'}: {predictions.tonic_note}
                </div>
              )}
            </div>
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

          {feedback === null && (
            <div style={styles.feedbackCard}>
              <div style={styles.feedbackQuestion}>Was this correct?</div>
              <div style={styles.feedbackBtns}>
                <button style={styles.feedbackYes} onClick={() => submitFeedback(topRaga, true)}>Yes</button>
                <button style={styles.feedbackNo} onClick={() => setFeedback('pending')}>No</button>
                <button style={styles.feedbackSkip} onClick={() => setFeedback('submitted')}>Skip</button>
              </div>
            </div>
          )}

          {feedback === 'pending' && (
            <div style={styles.feedbackCard}>
              <div style={styles.feedbackQuestion}>What raga was it?</div>
              <div style={styles.dropdownContainer}>
                <input
                  style={styles.searchInput}
                  placeholder="Search ragas..."
                  value={searchQuery}
                  onChange={e => { setSearchQuery(e.target.value); setShowDropdown(true) }}
                  onFocus={() => setShowDropdown(true)}
                />
                {showDropdown && (
                  <div style={styles.dropdownList}>
                    {filteredRagas.length > 0
                      ? filteredRagas.map(r => (
                          <div key={r} style={styles.dropdownItem}
                            onMouseDown={() => { setSelectedRaga(r); setSearchQuery(r); setShowDropdown(false) }}>
                            {r}
                          </div>
                        ))
                      : <div style={styles.dropdownEmpty}>Not found — we're working on adding more ragas!</div>
                    }
                  </div>
                )}
              </div>
              {selectedRaga && (
                <button style={styles.feedbackYes} onClick={() => submitFeedback(selectedRaga, false)}>
                  Submit
                </button>
              )}
            </div>
          )}

          {feedback === 'submitted' && (
            <div style={styles.feedbackThanks}>Thanks for the feedback!</div>
          )}

          <button style={styles.resetBtn} onClick={reset}>Try another</button>
        </>
      )}

      {error && <div style={styles.error}>{error}</div>}

      <div style={styles.about}>
        <div style={styles.aboutDivider} />
        <div style={styles.aboutContent}>
          <div style={styles.aboutText}>
            Built by <strong>Pratham Aithal</strong>, a high school student at Rock Hill High School in Frisco, TX (PISD), connecting a love of Carnatic vocal music with AI and machine learning.
          </div>
          <div style={styles.aboutLinks}>
            <a href="https://github.com/Smashgod23/raga-identifier" target="_blank" style={styles.aboutLink}>GitHub</a>
            <span style={styles.aboutLinkDot}>·</span>
            <a href="mailto:theprathamaithal@gmail.com" style={styles.aboutLink}>theprathamaithal@gmail.com</a>
          </div>
        </div>
      </div>
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
  youtubeCard: { background: '#fff', border: '1px solid #e8e2da', borderRadius: 12, padding: '20px', marginBottom: 12 },
  youtubeHeader: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 },
  youtubeIcon: { width: 40, height: 40, borderRadius: 10, background: '#fdf0eb', border: '1.5px solid #e8d4c4', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16, color: '#c4826a', flexShrink: 0 },
  youtubeInputRow: { display: 'flex', gap: 8 },
  youtubeInput: { flex: 1, padding: '10px 14px', borderRadius: 8, border: '1px solid #e8e2da', fontSize: 14, background: '#faf8f5', outline: 'none', fontFamily: "'DM Sans', sans-serif" },
  youtubeBtn: { padding: '10px 20px', borderRadius: 8, background: '#c4826a', border: 'none', color: '#fff', fontSize: 13, fontWeight: 500, cursor: 'pointer', whiteSpace: 'nowrap' },
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
  confidenceTag: { display: 'inline-block', background: '#eef6ee', color: '#5a8a5a', fontSize: 12, fontWeight: 500, padding: '3px 10px', borderRadius: 20 },
  tagRow: { display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 },
  saTag: { display: 'inline-block', background: '#f5f2ee', color: '#9a9082', fontSize: 12, fontWeight: 500, padding: '3px 10px', borderRadius: 20 },
  saTagSet: { display: 'inline-block', background: '#fdf0eb', color: '#c4826a', fontSize: 12, fontWeight: 500, padding: '3px 10px', borderRadius: 20 },
  tonicPicker: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, background: '#fff', border: '1px solid #e8e2da', borderRadius: 12, padding: '14px 18px', marginBottom: 12 },
  tonicLabel: { fontSize: 14, fontWeight: 500, color: '#3c3530', marginBottom: 2 },
  tonicHint: { fontSize: 11, color: '#b0a898', fontWeight: 400, marginLeft: 6 },
  tonicDesc: { fontSize: 11, color: '#9a9082', lineHeight: 1.4 },
  tonicSelect: { padding: '8px 12px', borderRadius: 8, border: '1px solid #e8e2da', fontSize: 13, background: '#faf8f5', color: '#3c3530', outline: 'none', fontFamily: "'DM Sans', sans-serif", cursor: 'pointer', minWidth: 150 },
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
  feedbackCard: { background: '#fff', border: '1px solid #e8e2da', borderRadius: 12, padding: '20px 24px', marginBottom: 12 },
  feedbackQuestion: { fontSize: 14, color: '#3c3530', marginBottom: 14, fontFamily: 'Georgia, serif' },
  feedbackBtns: { display: 'flex', gap: 8 },
  feedbackYes: { padding: '8px 20px', borderRadius: 8, background: '#eef6ee', border: '1px solid #c8dfc8', color: '#5a8a5a', fontSize: 13, cursor: 'pointer' },
  feedbackNo: { padding: '8px 20px', borderRadius: 8, background: '#fdf8f6', border: '1px solid #e8d4c4', color: '#c4826a', fontSize: 13, cursor: 'pointer' },
  feedbackSkip: { padding: '8px 20px', borderRadius: 8, background: '#f5f2ee', border: '1px solid #e0dbd4', color: '#9a9082', fontSize: 13, cursor: 'pointer' },
  dropdownContainer: { position: 'relative', marginBottom: 12 },
  searchInput: { width: '100%', padding: '10px 14px', borderRadius: 8, border: '1px solid #e8e2da', fontSize: 14, background: '#faf8f5', outline: 'none' },
  dropdownList: { position: 'absolute', top: '100%', left: 0, right: 0, background: '#fff', border: '1px solid #e8e2da', borderRadius: 8, maxHeight: 200, overflowY: 'auto', zIndex: 100, marginTop: 4 },
  dropdownItem: { padding: '10px 14px', fontSize: 14, color: '#3c3530', cursor: 'pointer', fontFamily: 'Georgia, serif' },
  dropdownEmpty: { padding: '10px 14px', fontSize: 13, color: '#9a9082', fontStyle: 'italic' },
  feedbackThanks: { textAlign: 'center', fontSize: 13, color: '#5a8a5a', padding: '16px', background: '#eef6ee', borderRadius: 10, marginBottom: 12 },
  resetBtn: { width: '100%', padding: '12px', borderRadius: 10, background: '#fff', border: '1px solid #e8e2da', fontSize: 14, color: '#9a9082', cursor: 'pointer', marginTop: 8 },
  error: { background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 10, padding: '12px 16px', fontSize: 13, color: '#b91c1c', marginTop: 12 },
  about: { marginTop: 40 },
  aboutDivider: { height: 1, background: '#e0dbd4', marginBottom: 24 },
  aboutContent: { textAlign: 'center' },
  aboutText: { fontSize: 13, color: '#9a9082', lineHeight: 1.7, marginBottom: 12, maxWidth: 480, margin: '0 auto 12px' },
  aboutLinks: { display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 },
  aboutLink: { fontSize: 13, color: '#c4826a', textDecoration: 'none' },
  aboutLinkDot: { color: '#c8c0b4', fontSize: 13 },
}

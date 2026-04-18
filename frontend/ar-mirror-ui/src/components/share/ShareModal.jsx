/**
 * ShareModal.jsx
 * Shows the shareable AR mirror snapshot with:
 *  - Web Share API (native image share to WhatsApp/Instagram/etc on mobile)
 *  - WhatsApp desktop fallback (download image → open WhatsApp)
 *  - X (Twitter) text share
 *  - Copy Link button
 *  - Download snapshot button
 *  - QR code (when backend is online)
 *  - NFC copy URI
 *  - 15-minute countdown timer
 */
import { motion } from 'framer-motion';
import QRCode from 'qrcode';
import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './ShareModal.module.css';

// ── Countdown badge ──────────────────────────────────────────────────────────

function CountdownBadge({ expiresAt }) {
  const [remaining, setRemaining] = useState(0);

  useEffect(() => {
    const update = () => {
      const nowSecs = Date.now() / 1000;
      const secs = Math.max(0, Math.round(expiresAt - nowSecs));
      setRemaining(secs);
    };
    update();
    const iv = setInterval(update, 1000);
    return () => clearInterval(iv);
  }, [expiresAt]);

  const m = String(Math.floor(remaining / 60)).padStart(2, '0');
  const s = String(remaining % 60).padStart(2, '0');

  return (
    <span className={styles.expireBadge}>
      ⏱ Expires in {m}:{s}
    </span>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Convert base64 JPEG string → File object (for Web Share API) */
async function b64ToFile(b64, filename = 'ar-mirror-tryon.jpg') {
  const res = await fetch(`data:image/jpeg;base64,${b64}`);
  const blob = await res.blob();
  return new File([blob], filename, { type: 'image/jpeg' });
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ShareModal({ shareData, offlineError, onClose }) {
  const { qr_b64, qr_content, preview_b64, share_url, nfc_payload, expires_at } = shareData;

  const [copied, setCopied]         = useState(false);
  const [nfcCopied, setNfcCopied]   = useState(false);
  const [waStatus, setWaStatus]     = useState('idle'); // 'idle' | 'sharing' | 'downloaded'
  const [nativeShare, setNativeShare] = useState(false);   // whether Web Share API worked
  const [clientQr, setClientQr]       = useState(null);  // browser-generated QR data URL

  const qrSrc      = qr_b64      ? `data:image/png;base64,${qr_b64}`      : clientQr;
  const previewSrc = preview_b64 ? `data:image/jpeg;base64,${preview_b64}` : null;

  // What the QR should encode: image data URL if provided, else share URL
  const qrTextToEncode = qr_content || share_url;

  // Generate QR code in the browser when the backend hasn't provided one
  useEffect(() => {
    if (qr_b64) return; // server already gave us a QR
    let cancelled = false;
    QRCode.toDataURL(qrTextToEncode, {
      width: 240,
      margin: 1,
      color: { dark: '#0f172a', light: '#ffffff' },
      errorCorrectionLevel: 'M',
    }).then((dataUrl) => {
      if (!cancelled) setClientQr(dataUrl);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [qr_b64, qrTextToEncode]);

  // ── WhatsApp: try native Web Share first, fall back to download + wa.me ──
  const handleWhatsApp = useCallback(async () => {
    if (waStatus === 'sharing') return;
    setWaStatus('sharing');

    const caption = '👕 Check out my virtual try-on with Zyro AR Mirror! 🪞✨';

    // Mobile path: Web Share API with image file
    if (preview_b64 && navigator.canShare) {
      try {
        const file = await b64ToFile(preview_b64);
        if (navigator.canShare({ files: [file] })) {
          await navigator.share({ files: [file], title: 'My Virtual Try-On', text: caption });
          setNativeShare(true);
          setWaStatus('idle');
          return;
        }
      } catch (err) {
        // User cancelled or API failed — fall through to desktop path
        if (err.name === 'AbortError') { setWaStatus('idle'); return; }
      }
    }

    // Desktop  / no Web Share: download the image then open WhatsApp with text
    if (preview_b64) {
      // Trigger image download so user can attach it
      const a = document.createElement('a');
      a.href = previewSrc;
      a.download = 'ar-mirror-tryon.jpg';
      a.click();
      // Short delay so download starts before opening a new tab
      await new Promise(r => setTimeout(r, 600));
      setWaStatus('downloaded');
      setTimeout(() => setWaStatus('idle'), 4000);
    }

    // Open WhatsApp with pre-filled text
    window.open(
      `https://wa.me/?text=${encodeURIComponent(caption + '\n' + share_url)}`,
      '_blank',
    );
  }, [preview_b64, previewSrc, share_url, waStatus]);

  // ── X / Twitter ──────────────────────────────────────────────────────────
  const handleX = useCallback(async () => {
    // Try native image share if available
    if (preview_b64 && navigator.canShare) {
      try {
        const file = await b64ToFile(preview_b64);
        if (navigator.canShare({ files: [file] })) {
          await navigator.share({ files: [file], title: 'My Virtual Try-On', text: 'My virtual try-on result! 🪞✨ #ZyroAR' });
          return;
        }
      } catch (err) {
        if (err.name === 'AbortError') return;
      }
    }
    // Desktop fallback: Twitter intent
    window.open(
      `https://twitter.com/intent/tweet?text=${encodeURIComponent('My virtual try-on result! 🪞✨ #ZyroAR')}&url=${encodeURIComponent(share_url)}`,
      '_blank',
    );
  }, [preview_b64, share_url]);

  // ── Copy link ─────────────────────────────────────────────────────────────
  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(share_url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch { /* unavailable */ }
  };

  const handleCopyNfc = async () => {
    try {
      await navigator.clipboard.writeText(nfc_payload || share_url);
      setNfcCopied(true);
      setTimeout(() => setNfcCopied(false), 2500);
    } catch { /* unavailable */ }
  };

  // ── Download ──────────────────────────────────────────────────────────────
  const handleDownload = () => {
    if (!preview_b64) return;
    const a = document.createElement('a');
    a.href = previewSrc;
    a.download = 'ar-mirror-tryon.jpg';
    a.click();
  };

  // ── Escape key ────────────────────────────────────────────────────────────
  useEffect(() => {
    const h = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  // WhatsApp button label
  const waLabel = waStatus === 'sharing'
    ? 'Opening…'
    : waStatus === 'downloaded'
      ? '📎 Image saved — attach in WhatsApp!'
      : 'WhatsApp';

  return (
    <motion.div
      className={styles.backdrop}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <motion.div
        className={styles.modal}
        initial={{ opacity: 0, scale: 0.92, y: 40 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        transition={{ type: 'spring', stiffness: 240, damping: 26 }}
      >
        {/* Header */}
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <span className={styles.icon}>🪞</span>
            <div>
              <h2 className={styles.title}>Share Your Try-On</h2>
              <p className={styles.subtitle}>
                {navigator.canShare ? 'Tap to share with image' : 'Share your look below'}
              </p>
            </div>
          </div>
          <button className={styles.closeBtn} onClick={onClose} aria-label="Close">✕</button>
        </div>

        {/* Offline warning */}
        {offlineError && (
          <div className={styles.offlineBanner}>
            ⚠️ {offlineError}
          </div>
        )}

        {/* Downloaded hint */}
        {waStatus === 'downloaded' && (
          <div className={styles.downloadHint}>
            📎 Image downloaded! Open WhatsApp → tap the attach (📎) button → select the downloaded photo.
          </div>
        )}

        {/* Expire badge */}
        <div className={styles.expireRow}>
          <CountdownBadge expiresAt={expires_at} />
        </div>

        {/* Main grid */}
        <div className={styles.grid}>

          {/* Left — QR + NFC */}
          <div className={styles.leftPanel}>
            <p className={styles.sectionLabel}>📱 Scan to view on phone</p>
            {qrSrc ? (
              <motion.div
                className={styles.qrWrap}
                initial={{ opacity: 0, scale: 0.85 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 }}
              >
                <img src={qrSrc} alt="QR Code" className={styles.qr} />
              </motion.div>
            ) : (
              <div className={styles.qrPlaceholder}>
                <p style={{ color: '#38bdf8' }}>⏳ Generating QR…</p>
              </div>
            )}

            {/* NFC */}
            <div className={styles.nfcPanel}>
              <div className={styles.nfcHeader}>
                <span className={styles.nfcIcon}>📡</span>
                <strong>NFC Tap</strong>
              </div>
              <p className={styles.nfcDesc}>
                Copy the URI below and write to an NFC tag using any NFC writer app (e.g. NFC Tools).
              </p>
              <div className={styles.nfcUri}>
                <code>{nfc_payload || share_url}</code>
              </div>
              <button className={styles.nfcCopyBtn} onClick={handleCopyNfc}>
                {nfcCopied ? '✓ Copied!' : 'Copy NFC URI'}
              </button>
            </div>
          </div>

          {/* Right — Preview + Social */}
          <div className={styles.rightPanel}>

            {/* Snapshot */}
            {previewSrc && (
              <div className={styles.previewWrap}>
                <img src={previewSrc} alt="Try-on snapshot" className={styles.preview} />
                <div className={styles.previewBadge}>LIVE CAPTURE</div>
              </div>
            )}

            {/* Social share tip for desktop */}
            {!navigator.canShare && preview_b64 && (
              <div className={styles.desktopTip}>
                💡 On mobile, WhatsApp shares the image directly. On desktop, the image will be saved first — then attach it in WhatsApp.
              </div>
            )}

            <p className={styles.sectionLabel}>🌐 Share to social</p>
            <div className={styles.socialRow}>

              {/* WhatsApp — Web Share on mobile, download+open on desktop */}
              <button
                className={`${styles.btn} ${styles.btnWhatsapp} ${waStatus !== 'idle' ? styles.btnLoading : ''}`}
                onClick={handleWhatsApp}
                disabled={waStatus === 'sharing'}
              >
                <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893A11.817 11.817 0 0020.885 3.488"/>
                </svg>
                {waLabel}
              </button>

              {/* X/Twitter */}
              <button
                className={`${styles.btn} ${styles.btnX}`}
                onClick={handleX}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.746l7.73-8.835L1.254 2.25H8.08l4.259 5.631zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                </svg>
                X (Twitter)
              </button>

              <button className={`${styles.btn} ${styles.btnCopy}`} onClick={handleCopyLink}>
                {copied ? '✓ Copied!' : '🔗 Copy Link'}
              </button>
            </div>

            {/* Download */}
            {previewSrc && (
              <button className={styles.downloadBtn} onClick={handleDownload}>
                ⬇ Download Snapshot
              </button>
            )}

            {/* Share URL */}
            <div className={styles.urlRow}>
              <span className={styles.urlLabel}>Share URL</span>
              <code className={styles.urlText}>{share_url}</code>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

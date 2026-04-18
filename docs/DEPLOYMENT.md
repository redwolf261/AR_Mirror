# Production Deployment Checklist

## Pre-Deployment Validation

### 1. Functional Testing
- [ ] Test on 10+ diverse body types
- [ ] Validate measurements against tape measure (±2cm tolerance)
- [ ] Test all failure modes trigger correctly
- [ ] Verify log files generate properly
- [ ] Test offline operation (no network)
- [ ] Test garment database update mechanism

### 2. Performance Testing
- [ ] Frame rate ≥15 FPS on target devices (Redmi 9A, Realme C11)
- [ ] Memory usage <150MB sustained
- [ ] No memory leaks after 30min session
- [ ] App startup time <3 seconds
- [ ] Time to first measurement <10 seconds

### 3. Edge Case Testing
- [ ] Very bright lighting (outdoors)
- [ ] Very dim lighting (evening, no flash)
- [ ] Occlusions (hands on hips, phone in hand)
- [ ] Multiple people in frame
- [ ] Patterned clothing (stripes, busy prints)
- [ ] Background clutter
- [ ] Camera movement/shake

### 4. Database Validation
- [ ] All SKUs have complete measurements
- [ ] Size labels consistent across categories
- [ ] No duplicate SKUs
- [ ] Measurements within valid ranges

## Android Build Configuration

### APK Optimization
```gradle
android {
    buildTypes {
        release {
            minifyEnabled true
            shrinkResources true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt')
        }
    }
    
    splits {
        abi {
            enable true
            reset()
            include 'armeabi-v7a', 'arm64-v8a'
            universalApk false
        }
    }
}
```

### Permissions (AndroidManifest.xml)
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

## Monitoring Setup

### Key Metrics to Track
1. **Success Rate:** successful_measurements / total_attempts
2. **Failure Breakdown:** count by failure_type
3. **Average Session Duration:** time from app start to fit decision
4. **Return Rate:** returned_items / items_purchased_with_sizing
5. **Device Performance:** median_fps by device_model

### Alert Triggers
- Success rate drops below 70%
- Crash rate exceeds 2%
- Any single failure mode exceeds 40% of failures
- Return rate for sized purchases exceeds 12%

## Gradual Rollout Strategy

### Phase 1: Internal Testing (Week 1-2)
- Deploy to team devices
- 50+ manual tests
- Iterate on UX messaging

### Phase 2: Pilot Retailer (Week 3-6)
- Single store, 100 SKUs
- Supervised usage (staff assisted)
- Collect 500+ measurements
- Daily metric review

### Phase 3: Multi-Store Beta (Week 7-12)
- 5 stores, same retailer
- Unsupervised usage
- Collect 5000+ measurements
- Weekly metric review

### Phase 4: Full Rollout
- All stores in network
- Continuous monitoring
- Weekly data analysis

## Post-Deployment Data Analysis

### Weekly Tasks
1. Generate failure mode distribution
2. Compute measurement bias per demographic
3. Identify high-return SKUs
4. Update bias correction lookup tables

### Monthly Tasks
1. Train return prediction model
2. Generate size chart inconsistency report for vendors
3. Device performance profiling
4. UX improvement prioritization

## Support & Troubleshooting

### User-Facing Issues

**"App crashes on launch"**
- Check device has 2GB+ RAM
- Verify Android 7.0+
- Reinstall app

**"Cannot detect pose"**
- Improve lighting
- Stand 1.5-2m from camera
- Ensure full body in frame
- Remove obstructions

**"Measurements seem wrong"**
- Verify correct pose (frontal, straight)
- Check distance (target on screen)
- Retry in better lighting

### Developer Diagnostics

**Enable Debug Mode:**
```python
pipeline = SizingPipeline(garment_db_path, log_dir, debug=True)
```

Shows on-screen:
- Landmark coordinates
- Scale factor
- Raw measurements
- Processing time per frame

**Log Analysis:**
```bash
# Count failures by type
jq -r 'select(.event_type=="failure") | .data.reason' logs/*.jsonl | sort | uniq -c

# Average confidence scores
jq -r 'select(.event_type=="fit_result") | .data.measurements.confidence' logs/*.jsonl | awk '{sum+=$1; n++} END {print sum/n}'

# Return rate by SKU
# (requires linking with purchase/return data)
```

## Security Considerations

### Data Privacy
- No raw images stored
- User IDs hashed (SHA-256 + salt)
- Logs encrypted at rest (AES-256)
- HTTPS only for sync
- GDPR-compliant data retention

### App Security
- Certificate pinning for API calls
- No hardcoded credentials
- ProGuard obfuscation
- Root detection (warn user, continue)

## Maintenance Schedule

### Daily
- Monitor crash reports
- Check log sync status

### Weekly
- Review failure metrics
- Deploy bias correction updates (if needed)

### Monthly
- Update MediaPipe model (if new version)
- Refresh garment database
- Review return rate correlation

### Quarterly
- Comprehensive accuracy audit
- User satisfaction survey
- Device compatibility review

## Rollback Plan

### Trigger Conditions
- Crash rate >5%
- Success rate <50%
- Critical bug discovered

### Rollback Steps
1. Disable auto-updates
2. Deploy previous stable version
3. Investigate issue in staging
4. Fix and re-test
5. Gradual re-rollout

## Success Criteria for V1.0

**Technical:**
- ✅ 80%+ measurement success rate
- ✅ <2% crash rate
- ✅ 15+ FPS on target devices
- ✅ <100ms latency

**Business:**
- ✅ 20%+ reduction in size-related returns
- ✅ 10,000+ measurements collected
- ✅ 3 retailers onboarded
- ✅ Positive user feedback (>4.0/5.0)

**Data Moat:**
- ✅ Bias correction tables deployed
- ✅ Return prediction model trained
- ✅ 50+ SKUs flagged for size chart issues

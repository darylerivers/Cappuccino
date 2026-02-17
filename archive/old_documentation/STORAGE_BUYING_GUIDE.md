# Storage Buying Guide for Cappuccino Trading Project

**Date:** January 17, 2026
**Current Usage:** 187 GB (184 GB models, 2.3 GB databases, 1 GB logs/data)
**Current Trials:** 1,351 completed

---

## TL;DR - Recommendation

### Best Value: 4 TB Drive

**Specs:**
- **Capacity:** 4 TB
- **Cache:** 256 MB
- **Speed:** 7200 RPM (CMR, not SMR)
- **Use Case:** Data storage / NAS
- **Interface:** SATA 6Gb/s
- **Price Target:** $70-90

**Recommended Models:**
1. **Western Digital Red Plus 4TB** (~$85)
   - CMR technology (better for writes)
   - 256 MB cache
   - NAS-optimized, very reliable
   - 3-year warranty

2. **Seagate IronWolf 4TB** (~$80)
   - CMR, 256 MB cache
   - Designed for NAS/24x7 operation
   - 3-year warranty

3. **Toshiba N300 4TB** (~$75)
   - CMR, 256 MB cache
   - NAS-grade reliability
   - Slightly cheaper

**Why 4TB?**
- Current needs: 187 GB
- At 5,000 trials: ~680 GB
- With backups (3x rotation): ~2 TB
- Future experiments: +500 GB
- Total headroom: ~1.5 TB spare

---

## Your Current Storage Breakdown

### Project Usage (187 GB total)

```
train_results/    184 GB  (98%)  - Model checkpoints
databases/        2.3 GB  (1%)   - Optuna database
logs/             173 MB  (<1%)  - Training and trading logs
data/             117 MB  (<1%)  - Crypto price data
paper_trades/     1 MB    (<1%)  - Paper trading CSVs
backups/          271 MB  (<1%)  - Model backups
```

**Key insight:** 98% of storage is model checkpoints!

### Model Storage Details

- **Total models:** 3,619 actor.pth files
- **Model sizes:** 3-6 MB each
- **Average per trial:** ~136 MB (includes actor, critic, logs)
- **Storage per 1,000 trials:** ~136 GB

---

## Storage Projections

### Training Growth

| Trials | Storage Needed | Timeline |
|--------|----------------|----------|
| 1,351 (current) | 184 GB | Now |
| 2,000 | 272 GB | +2 days |
| 3,000 | 408 GB | +1 week |
| 5,000 | 680 GB | +2 weeks |
| 10,000 | 1.36 TB | +1 month |

### With Automated Backups

Assuming hourly backups with rotation:
- 24 hourly backups (most recent day)
- 7 daily backups (most recent week)
- 4 weekly backups (most recent month)

**Backup multiplier:** ~3x active storage

| Active Data | With Backups | Recommendation |
|-------------|--------------|----------------|
| 200 GB | 600 GB | 1 TB tight |
| 500 GB | 1.5 TB | 2 TB comfortable |
| 700 GB | 2.1 TB | 4 TB future-proof |
| 1 TB | 3 TB | 4 TB with headroom |

**At 5,000 trials:** 680 GB active + 2 TB backups = **2.7 TB needed**

**Recommendation:** 4 TB drive gives comfortable margin

---

## Drive Specifications Explained

### 1. Capacity: 4 TB ⭐

**Why 4TB over 2TB?**
- Current: 187 GB
- At scale (5k trials): 680 GB
- With backups: 2 TB
- Future experiments: +500 GB
- **Total needs: 2.5-3 TB**

**Why not 8TB+?**
- More expensive ($/TB not much better)
- Overkill for current project
- 4 TB is sweet spot

**Price difference:**
- 2 TB: ~$50 (too small)
- 4 TB: ~$80 (**best value**)
- 8 TB: ~$150 (unnecessary)

### 2. Cache: 256 MB ⭐

**What it does:**
- Buffer for frequently accessed data
- Speeds up reads/writes
- Important for database operations

**Why 256 MB?**
- Modern standard (64/128 MB is old)
- Better for Optuna database reads
- Faster model checkpoint saves
- Minimal price difference vs 128 MB

**Benchmarks:**
- 64 MB: Slow for large files
- 128 MB: OK, baseline
- 256 MB: **Fast, recommended**
- 512 MB: Marginal gains, expensive

### 3. Speed: 7200 RPM ⭐

**Why 7200 RPM over 5400 RPM?**
- 33% faster sequential read/write
- Better for backup operations
- Faster model loading/saving
- Worth the $5-10 premium

**Speed comparison:**
- 5400 RPM: ~120 MB/s (slow)
- 7200 RPM: ~160 MB/s (**faster**)
- SSD: ~500 MB/s (expensive, overkill for backup)

**Use case fit:**
- Backups: 7200 RPM is perfect
- Active training: Already on NVMe SSD ✓
- Archives: 7200 RPM sufficient

### 4. Technology: CMR (Not SMR) ⭐⭐⭐

**CRITICAL: Avoid SMR drives for backups!**

**CMR (Conventional Magnetic Recording):**
- ✅ Fast writes
- ✅ Good for backups
- ✅ Reliable for databases
- ✅ No performance degradation

**SMR (Shingled Magnetic Recording):**
- ❌ Slow writes (rewrites adjacent tracks)
- ❌ Terrible for backups
- ❌ Can fail during large writes
- ❌ Cheaper but awful for this use case

**How to identify:**
- Look for "CMR" or "NAS" in specs
- WD Red **Plus** = CMR ✓
- WD Red (no plus) = SMR ❌
- Seagate IronWolf = CMR ✓
- Seagate Barracuda = Often SMR ❌

**For backups, CMR is mandatory!**

### 5. Use Case: NAS/Data Storage

**Drive categories:**

**Desktop/Consumer (Barracuda, Blue):**
- Cheaper
- Less reliable
- 1-2 year warranty
- OK for non-critical data

**NAS/Enterprise (Red Plus, IronWolf):**
- More reliable
- Designed for 24/7 operation
- 3-year warranty
- **Better for this project**

**Surveillance (Purple, SkyHawk):**
- Optimized for video writes
- Not ideal for random I/O
- Skip these

**Why NAS-grade?**
- You have automated backups running 24/7
- Need reliability for long-term storage
- Worth $10-15 premium
- Better warranty

---

## Recommended Drives (January 2026)

### 1. Western Digital Red Plus 4TB ⭐ TOP PICK

**Price:** ~$85

**Specs:**
- Capacity: 4 TB
- Cache: 256 MB
- Speed: 7200 RPM
- Tech: CMR
- Warranty: 3 years
- MTBF: 1 million hours

**Pros:**
- Excellent reliability
- CMR confirmed (not SMR)
- Designed for NAS/24x7
- Good customer support

**Cons:**
- Slightly more expensive than competitors
- Not the fastest (but sufficient)

**Where to buy:**
- Amazon: ~$85
- Newegg: ~$83
- B&H Photo: ~$85

---

### 2. Seagate IronWolf 4TB ⭐ BEST VALUE

**Price:** ~$80

**Specs:**
- Capacity: 4 TB
- Cache: 256 MB
- Speed: 7200 RPM
- Tech: CMR
- Warranty: 3 years
- MTBF: 1 million hours

**Pros:**
- Slightly cheaper than WD Red Plus
- Excellent for NAS
- CMR technology
- Includes Rescue Data Recovery (2 years)

**Cons:**
- Some users report higher failure rates (debated)
- Slightly louder than WD

**Where to buy:**
- Amazon: ~$80
- Newegg: ~$78
- B&H Photo: ~$80

---

### 3. Toshiba N300 4TB 💰 BUDGET OPTION

**Price:** ~$75

**Specs:**
- Capacity: 4 TB
- Cache: 256 MB
- Speed: 7200 RPM
- Tech: CMR
- Warranty: 3 years

**Pros:**
- Cheapest of the three
- CMR technology
- NAS-grade reliability
- Good performance

**Cons:**
- Less well-known brand
- Fewer reviews/user data
- Availability varies

**Where to buy:**
- Amazon: ~$75
- Newegg: ~$73

---

### Budget Alternative: 2 TB (~$50)

**If $80 is too much:**
- WD Red Plus 2TB: ~$55
- Seagate IronWolf 2TB: ~$50

**Limitations:**
- Only ~1 TB free after 5,000 trials + backups
- Will need upgrade in 6-12 months
- Not future-proof

**Recommendation:** Save $25-30 and get 4TB. Much better value long-term.

---

## What to Avoid

### ❌ Avoid These:

**1. SMR Drives**
- WD Red (without "Plus")
- Seagate Barracuda (most models)
- Any "Archive" drives
- Terrible for backups

**2. 5400 RPM Drives**
- Too slow for backup operations
- Not worth the $5 savings

**3. External USB Drives**
- OK for occasional backups
- Not good for automated/frequent backups
- Usually SMR inside
- USB overhead

**4. Used/Refurbished Drives**
- No warranty
- Unknown usage history
- Not worth the risk for data storage

**5. < 2 TB Capacity**
- Will outgrow quickly
- Poor $/GB value
- Need upgrade soon

---

## Buying Checklist

Before purchasing, verify:

- [ ] **Capacity:** 4 TB (or 2 TB minimum)
- [ ] **Cache:** 256 MB (128 MB acceptable)
- [ ] **Speed:** 7200 RPM (not 5400 RPM)
- [ ] **Technology:** CMR (not SMR) - Check specs!
- [ ] **Use Case:** NAS or Enterprise (not Desktop)
- [ ] **Warranty:** 3 years minimum
- [ ] **Price:** $75-90 for 4TB
- [ ] **Seller:** Reputable (Amazon, Newegg, B&H)

---

## Setup After Purchase

### 1. Format and Mount

```bash
# List drives
lsblk

# Format as ext4 (assuming /dev/sdb)
sudo mkfs.ext4 /dev/sdb1

# Create mount point
sudo mkdir -p /mnt/backup

# Mount
sudo mount /dev/sdb1 /mnt/backup

# Add to /etc/fstab for auto-mount
echo '/dev/sdb1 /mnt/backup ext4 defaults 0 2' | sudo tee -a /etc/fstab
```

### 2. Setup Automated Backups

Create backup script (Tool #4 from our earlier list):
```bash
# Backup models
rsync -avh --delete /opt/user-data/experiment/cappuccino/train_results/ /mnt/backup/models/

# Backup databases
rsync -avh /opt/user-data/experiment/cappuccino/databases/ /mnt/backup/databases/

# Backup logs
rsync -avh /opt/user-data/experiment/cappuccino/logs/ /mnt/backup/logs/

# Rotate backups (keep 24 hourly, 7 daily, 4 weekly)
```

### 3. Test Restore

```bash
# Verify backups are readable
ls -lh /mnt/backup/

# Test restore one model
cp /mnt/backup/models/cwd_tests/trial_861_1h/actor.pth /tmp/test.pth
```

---

## Price Summary (January 2026)

| Drive | Capacity | Price | $/TB | Use Case |
|-------|----------|-------|------|----------|
| Toshiba N300 | 4 TB | $75 | $18.75 | Budget |
| Seagate IronWolf | 4 TB | $80 | $20.00 | Best Value |
| WD Red Plus | 4 TB | $85 | $21.25 | Most Reliable |
| WD Red Plus | 2 TB | $55 | $27.50 | Tight Budget |

**Best bang for buck:** Seagate IronWolf 4TB ($80)
**Most reliable:** WD Red Plus 4TB ($85)
**If budget constrained:** Toshiba N300 4TB ($75)

---

## Long-Term Storage Costs

### 5-Year Ownership

**4 TB Drive:**
- Purchase: $80
- Power (24/7): ~$15/year = $75 over 5 years
- **Total: ~$155 over 5 years**
- Storage per year: $31

**Alternative (Cloud Storage):**
- 1 TB cloud: ~$60/year
- 2 TB needed: ~$120/year
- **5 years: $600**

**HDD is 75% cheaper than cloud over 5 years!**

---

## FAQ

**Q: Can I use an external USB drive?**
A: Not recommended for automated backups. Internal SATA is much better. External is OK for occasional manual backups.

**Q: Should I get 2x 2TB instead of 1x 4TB?**
A: No. 1x 4TB is simpler, uses less power, one failure point instead of two.

**Q: Do I need RAID?**
A: Not necessary. Backups of NVMe SSD is sufficient. RAID adds complexity without much benefit for this use case.

**Q: What about NVMe SSD for backups?**
A: Overkill and expensive. HDD is perfect for backups (sequential writes). Save NVMe for active training (already have).

**Q: 4TB seems like a lot. Can I go smaller?**
A: 2TB will work but you'll outgrow it in 6-12 months. 4TB gives years of headroom. Worth the extra $25.

**Q: How long will a 4TB drive last?**
A: NAS-grade drives: 5-7 years typical, up to 10+ years possible. MTBF is 1 million hours (114 years theoretical).

---

## Conclusion

**Buy this:** Seagate IronWolf 4TB ($80) or WD Red Plus 4TB ($85)

**Why:**
- 4 TB handles current + future (5,000+ trials)
- 256 MB cache, 7200 RPM, CMR tech
- NAS-grade reliability
- 3-year warranty
- Best value for your use case

**Where:** Amazon, Newegg, or B&H Photo (all reputable)

**When:** Buy now before prices change or drive models get discontinued

**Setup:** Takes 30 minutes to format, mount, and configure backups

---

**Your storage needs:** 187 GB now → 2-3 TB at scale
**Recommended:** 4 TB drive with 256 MB cache, 7200 RPM, CMR
**Price target:** $75-85
**Best buy:** Seagate IronWolf 4TB ($80) ⭐

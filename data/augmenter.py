"""
Targeted Data Augmentation (TA) for PII span-level diversification.

Operates directly on annotated PII spans via label-preserving substitution,
introducing diverse formatting patterns and controlled character-level
perturbations while leaving surrounding context unchanged.

Uses sequential offset tracking to maintain precise span boundaries after
augmentation with varying-length replacements.
"""
import re
import copy
import random
import string
from collections import Counter
from typing import List, Dict, Optional, Callable
from utils.logger import setup_logger

logger = setup_logger("TargetedAugmenter")

# =========================================================================
# PER-LABEL SPAN TRANSFORMERS
# Each function receives the original span text and returns a variant.
# Returning None means "skip this span" (no good augmentation available).
# =========================================================================

def _rand_digit_str(n):
    return "".join(random.choices(string.digits, k=n))

# ---- NAME / FIRST_NAME / LAST_NAME ----
def _aug_name(value: str) -> Optional[str]:
    strategies = []
    parts = value.split()
    if len(parts) >= 2:
        # Swap order: "John Smith" -> "Smith, John"
        strategies.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
        # Abbreviate first: "John Smith" -> "J. Smith"
        strategies.append(f"{parts[0][0]}. {' '.join(parts[1:])}")
        # Abbreviate last: "John Smith" -> "John S."
        strategies.append(f"{' '.join(parts[:-1])} {parts[-1][0]}.")
    # Full uppercase
    strategies.append(value.upper())
    # Full lowercase
    strategies.append(value.lower())
    # Title case (might already be, but ensures diversity)
    strategies.append(value.title())
    # Filter out variants identical to original
    strategies = [s for s in strategies if s != value]
    return random.choice(strategies) if strategies else None


# ---- EMAIL ----
_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "protonmail.com", "mail.com", "icloud.com", "aol.com",
    "zoho.com", "fastmail.com", "example.org", "example.net",
]

def _aug_email(value: str) -> Optional[str]:
    m = re.match(r'^([^@]+)@([^@]+)$', value.strip())
    if not m:
        return None
    local, domain = m.group(1), m.group(2)
    strategies = []
    # Swap domain
    other_domains = [d for d in _EMAIL_DOMAINS if d != domain]
    if other_domains:
        strategies.append(f"{local}@{random.choice(other_domains)}")
    # Add/remove dots in local part
    if '.' in local:
        strategies.append(f"{local.replace('.', '_')}@{domain}")
    else:
        mid = len(local) // 2
        if mid > 0:
            strategies.append(f"{local[:mid]}.{local[mid:]}@{domain}")
    # Case variation
    strategies.append(f"{local.upper()}@{domain}")
    strategies.append(f"{local.lower()}@{domain}")
    strategies = [s for s in strategies if s != value]
    return random.choice(strategies) if strategies else None


# ---- PHONE / PHONE_NUMBER / FAX_NUMBER ----
def _extract_digits(value: str) -> str:
    return re.sub(r'\D', '', value)

def _aug_phone(value: str) -> Optional[str]:
    digits = _extract_digits(value)
    if len(digits) < 7:
        return None

    # Detect if there's a country code prefix in original
    has_plus = value.strip().startswith('+')
    
    # Try to split into country code + national number
    # Common lengths: 10 (US), 11 (with country), 12-13 (intl)
    strategies = []
    
    if len(digits) == 10:
        a, b, c = digits[:3], digits[3:6], digits[6:]
        strategies.extend([
            f"({a}) {b}-{c}",
            f"{a}-{b}-{c}",
            f"{a}.{b}.{c}",
            f"{a} {b} {c}",
            f"+1 {a}-{b}-{c}",
            f"+1 ({a}) {b}-{c}",
        ])
    elif len(digits) >= 11:
        cc = digits[:-10]
        nat = digits[-10:]
        a, b, c = nat[:3], nat[3:6], nat[6:]
        strategies.extend([
            f"+{cc} {a}-{b}-{c}",
            f"+{cc} ({a}) {b}-{c}",
            f"+{cc} {a} {b} {c}",
            f"+{cc} {a}.{b}.{c}",
            f"({a}) {b}-{c}",
            f"{a}-{b}-{c}",
        ])
    else:
        # Generic: group digits differently
        mid = len(digits) // 2
        strategies.extend([
            f"{digits[:mid]}-{digits[mid:]}",
            f"{digits[:mid]} {digits[mid:]}",
            digits,  # no separators
        ])
    strategies = [s for s in strategies if s != value]
    return random.choice(strategies) if strategies else None


# ---- DATE / DATE_OF_BIRTH ----
_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def _parse_date_components(value: str):
    """Try to extract (year, month, day) from common date formats."""
    value = value.strip()
    # ISO: 2024-01-15
    m = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', value)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    # US: 01/15/2024 or 01-15-2024
    m = re.match(r'^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$', value)
    if m:
        return int(m.group(3)), int(m.group(1)), int(m.group(2))
    # Named: "15 March 2024" or "March 15, 2024"
    for i, name in enumerate(_MONTH_NAMES):
        # "15 March 2024"
        m = re.match(rf'^(\d{{1,2}})\s+{name}\s+(\d{{4}})$', value, re.IGNORECASE)
        if m:
            return int(m.group(2)), i + 1, int(m.group(1))
        # "March 15, 2024"
        m = re.match(rf'^{name}\s+(\d{{1,2}}),?\s+(\d{{4}})$', value, re.IGNORECASE)
        if m:
            return int(m.group(2)), i + 1, int(m.group(1))
    for i, abbr in enumerate(_MONTH_ABBR):
        m = re.match(rf'^(\d{{1,2}})\s+{abbr}\s+(\d{{4}})$', value, re.IGNORECASE)
        if m:
            return int(m.group(2)), i + 1, int(m.group(1))
        m = re.match(rf'^{abbr}\s+(\d{{1,2}}),?\s+(\d{{4}})$', value, re.IGNORECASE)
        if m:
            return int(m.group(2)), i + 1, int(m.group(1))
    return None

def _aug_date(value: str) -> Optional[str]:
    parsed = _parse_date_components(value)
    if not parsed:
        return None
    y, mo, d = parsed
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    strategies = [
        f"{y}-{mo:02d}-{d:02d}",           # ISO
        f"{mo:02d}/{d:02d}/{y}",            # US slash
        f"{d:02d}-{mo:02d}-{y}",            # EU dash
        f"{d} {_MONTH_NAMES[mo-1]} {y}",    # "15 March 2024"
        f"{_MONTH_NAMES[mo-1]} {d}, {y}",   # "March 15, 2024"
        f"{_MONTH_ABBR[mo-1]} {d}, {y}",    # "Mar 15, 2024"
        f"{d}/{mo:02d}/{y}",                # DD/MM/YYYY
        f"{y}.{mo:02d}.{d:02d}",            # Dotted ISO
    ]
    strategies = [s for s in strategies if s != value]
    return random.choice(strategies) if strategies else None


# ---- SSN ----
def _aug_ssn(value: str) -> Optional[str]:
    digits = _extract_digits(value)
    if len(digits) != 9:
        return None
    a, b, c = digits[:3], digits[3:5], digits[5:]
    strategies = [
        f"{a}-{b}-{c}",
        f"{a} {b} {c}",
        f"{a}.{b}.{c}",
        digits,  # no separator
    ]
    strategies = [s for s in strategies if s != value]
    return random.choice(strategies) if strategies else None


# ---- CREDIT_CARD / CREDIT_DEBIT_CARD ----
def _aug_credit_card(value: str) -> Optional[str]:
    digits = _extract_digits(value)
    if len(digits) < 13:
        return None
    # Standard 16-digit groupings
    if len(digits) == 16:
        strategies = [
            f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:]}",
            f"{digits[:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:]}",
            f"{digits[:4]}.{digits[4:8]}.{digits[8:12]}.{digits[12:]}",
            digits,
        ]
    else:
        mid = len(digits) // 2
        strategies = [
            f"{digits[:mid]} {digits[mid:]}",
            digits,
        ]
    strategies = [s for s in strategies if s != value]
    return random.choice(strategies) if strategies else None


# ---- URL ----
def _aug_url(value: str) -> Optional[str]:
    strategies = []
    v = value.strip()
    # Toggle http/https
    if v.startswith("https://"):
        strategies.append(v.replace("https://", "http://", 1))
    elif v.startswith("http://"):
        strategies.append(v.replace("http://", "https://", 1))
    # Toggle www
    if "://www." in v:
        strategies.append(v.replace("://www.", "://", 1))
    elif "://" in v and "://www." not in v:
        strategies.append(v.replace("://", "://www.", 1))
    # Trailing slash
    if v.endswith("/"):
        strategies.append(v.rstrip("/"))
    else:
        strategies.append(v + "/")
    strategies = [s for s in strategies if s != value]
    return random.choice(strategies) if strategies else None


# ---- ADDRESS / STREET_ADDRESS ----
_ADDR_ABBR = [
    ("Street", "St."), ("Avenue", "Ave."), ("Boulevard", "Blvd."),
    ("Drive", "Dr."), ("Court", "Ct."), ("Place", "Pl."),
    ("Lane", "Ln."), ("Road", "Rd."), ("Circle", "Cir."),
    ("Apartment", "Apt."), ("Suite", "Ste."), ("Highway", "Hwy."),
    ("Parkway", "Pkwy."), ("Square", "Sq."),
]

def _aug_address(value: str) -> Optional[str]:
    result = value
    changed = False
    for full, abbr in _ADDR_ABBR:
        # Try both directions: expand or abbreviate
        if abbr in result:
            result = result.replace(abbr, full, 1)
            changed = True
            break
        elif full in result:
            result = result.replace(full, abbr, 1)
            changed = True
            break
    if not changed:
        # Case variation
        strategies = [value.upper(), value.title()]
        strategies = [s for s in strategies if s != value]
        return random.choice(strategies) if strategies else None
    return result if result != value else None


# ---- IPV4 ----
def _aug_ipv4(value: str) -> Optional[str]:
    parts = value.strip().split('.')
    if len(parts) != 4:
        return None
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    strategies = [
        ".".join(f"{n:03d}" for n in nums),  # zero-padded
        ".".join(str(n) for n in nums),       # no padding
    ]
    strategies = [s for s in strategies if s != value.strip()]
    return random.choice(strategies) if strategies else None


# ---- USER_NAME ----
def _aug_username(value: str) -> Optional[str]:
    strategies = []
    if '_' in value:
        strategies.append(value.replace('_', '-'))
        strategies.append(value.replace('_', '.'))
        strategies.append(value.replace('_', ''))
    elif '-' in value:
        strategies.append(value.replace('-', '_'))
        strategies.append(value.replace('-', '.'))
    elif '.' in value:
        strategies.append(value.replace('.', '_'))
    strategies.append(value.lower())
    strategies.append(value.upper())
    # Append random digits
    strategies.append(value + str(random.randint(0, 99)))
    strategies = [s for s in strategies if s != value]
    return random.choice(strategies) if strategies else None


# ---- ACCOUNT_NUMBER / CUSTOMER_ID / EMPLOYEE_ID ----
def _aug_id_number(value: str) -> Optional[str]:
    digits = _extract_digits(value)
    non_digits = re.sub(r'\d', '', value)
    if len(digits) < 4:
        return None
    strategies = []
    # Regroup with different separators
    if '-' in value:
        strategies.append(value.replace('-', ' '))
        strategies.append(value.replace('-', ''))
    elif ' ' in value:
        strategies.append(value.replace(' ', '-'))
        strategies.append(value.replace(' ', ''))
    else:
        mid = len(digits) // 2
        strategies.append(f"{digits[:mid]}-{digits[mid:]}")
    # Zero-pad
    if len(digits) < 12:
        strategies.append(digits.zfill(12))
    strategies = [s for s in strategies if s != value]
    return random.choice(strategies) if strategies else None


# ---- PASSWORD ----
_LEET_MAP = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$', 't': '7', 'l': '1'}

def _aug_password(value: str) -> Optional[str]:
    strategies = []
    # Leet speak variant
    leet = list(value)
    changed = False
    for i, c in enumerate(leet):
        if c.lower() in _LEET_MAP and random.random() < 0.4:
            leet[i] = _LEET_MAP[c.lower()]
            changed = True
    if changed:
        strategies.append("".join(leet))
    # Case swap
    strategies.append(value.swapcase())
    # Append special char
    strategies.append(value + random.choice("!@#$%^&*"))
    strategies = [s for s in strategies if s != value]
    return random.choice(strategies) if strategies else None


# ---- TIME ----
def _aug_time(value: str) -> Optional[str]:
    # Try HH:MM or HH:MM:SS (24h)
    m = re.match(r'^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?$', value.strip(), re.IGNORECASE)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    sec = int(m.group(3)) if m.group(3) else None
    ampm = m.group(4)

    strategies = []
    if ampm:
        # Convert to 24h
        h24 = h % 12 + (12 if ampm.upper() == 'PM' else 0)
        if sec is not None:
            strategies.append(f"{h24:02d}:{mi:02d}:{sec:02d}")
        else:
            strategies.append(f"{h24:02d}:{mi:02d}")
    else:
        # Convert to 12h
        suffix = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        if sec is not None:
            strategies.append(f"{h12}:{mi:02d}:{sec:02d} {suffix}")
        else:
            strategies.append(f"{h12}:{mi:02d} {suffix}")
    strategies = [s for s in strategies if s != value.strip()]
    return random.choice(strategies) if strategies else None


# ---- GENERIC FALLBACK: char-level perturbation ----
_CONFUSABLES = {
    'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
    'A': 'А', 'E': 'Е', 'O': 'О', 'P': 'Р', 'C': 'С',
}

def _aug_generic_casing(value: str) -> Optional[str]:
    """Case-based augmentation usable for any text span."""
    if value != value.upper():
        return value.upper()
    if value != value.lower():
        return value.lower()
    return None


# =========================================================================
# LABEL → TRANSFORMER DISPATCH
# =========================================================================
_LABEL_AUGMENTERS: Dict[str, Callable] = {
    # Names
    "NAME": _aug_name,
    "FIRST_NAME": _aug_name,
    "LAST_NAME": _aug_name,
    # Contact
    "EMAIL": _aug_email,
    "PHONE": _aug_phone,
    "PHONE_NUMBER": _aug_phone,
    "FAX_NUMBER": _aug_phone,
    # Dates
    "DATE": _aug_date,
    "DATE_OF_BIRTH": _aug_date,
    # IDs
    "SSN": _aug_ssn,
    "NATIONAL_ID": _aug_ssn,
    "TAX_ID": _aug_ssn,
    "CREDIT_CARD": _aug_credit_card,
    "CREDIT_DEBIT_CARD": _aug_credit_card,
    # Network
    "URL": _aug_url,
    "IPV4": _aug_ipv4,
    "USER_NAME": _aug_username,
    # Address
    "ADDRESS": _aug_address,
    "STREET_ADDRESS": _aug_address,
    # IDs / codes
    "ACCOUNT_NUMBER": _aug_id_number,
    "CUSTOMER_ID": _aug_id_number,
    "EMPLOYEE_ID": _aug_id_number,
    "CERTIFICATE_LICENSE_NUMBER": _aug_id_number,
    # Auth
    "PASSWORD": _aug_password,
    # Time
    "TIME": _aug_time,
}


# =========================================================================
# CORE AUGMENTATION ENGINE
# =========================================================================

def _augment_single_record(record: Dict, aug_prob: float = 0.5) -> Optional[Dict]:
    """
    Create one augmented copy of a record by applying span-level
    transformations with sequential offset tracking.

    Returns None if no span was actually changed.
    """
    text = record["source_text"]
    masks = record.get("privacy_mask", [])
    if not masks:
        return None

    # Sort spans by start position for sequential offset tracking
    sorted_masks = sorted(enumerate(masks), key=lambda x: x[1].get("start", 0))

    new_text = text
    new_masks = copy.deepcopy(masks)
    cumulative_offset = 0
    any_changed = False

    for orig_idx, mask in sorted_masks:
        label = mask.get("label", "").strip().upper()
        start = mask.get("start", 0)
        end = mask.get("end", 0)
        orig_value = mask.get("value", text[start:end])

        # Decide whether to augment this span
        if random.random() > aug_prob:
            # No augmentation — just shift offsets
            new_masks[orig_idx]["start"] = start + cumulative_offset
            new_masks[orig_idx]["end"] = end + cumulative_offset
            continue

        # Find augmenter for this label
        augmenter = _LABEL_AUGMENTERS.get(label)
        if augmenter is None:
            # No augmenter for this label type — skip, don't add noise
            new_masks[orig_idx]["start"] = start + cumulative_offset
            new_masks[orig_idx]["end"] = end + cumulative_offset
            continue

        # Extract leading/trailing whitespaces from original value
        orig_len = len(orig_value)
        stripped_value = orig_value.lstrip()
        leading_spaces = orig_value[:orig_len - len(stripped_value)]
        stripped_value = stripped_value.rstrip()
        trailing_spaces = orig_value[len(leading_spaces) + len(stripped_value):]

        new_value = augmenter(stripped_value)
        if new_value is None or new_value == stripped_value:
            # No useful augmentation available
            new_masks[orig_idx]["start"] = start + cumulative_offset
            new_masks[orig_idx]["end"] = end + cumulative_offset
            continue

        # Restore original whitespaces to prevent token fusion
        new_value = leading_spaces + new_value + trailing_spaces

        # Apply substitution with offset tracking
        adj_start = start + cumulative_offset
        adj_end = end + cumulative_offset
        new_text = new_text[:adj_start] + new_value + new_text[adj_end:]

        length_diff = len(new_value) - (end - start)
        new_masks[orig_idx]["start"] = adj_start
        new_masks[orig_idx]["end"] = adj_start + len(new_value)
        new_masks[orig_idx]["value"] = new_value
        cumulative_offset += length_diff
        any_changed = True

    if not any_changed:
        return None

    return {
        "source_text": new_text,
        "privacy_mask": new_masks,
        "language": record.get("language", ""),
    }


def targeted_augmentation(
    train_data: List[Dict],
    augment_ratio: float = 0.3,
    minority_boost: int = 1,
    minority_threshold_pct: float = 0.10,
    aug_prob_per_span: float = 0.6,
    seed: int = 42,
) -> List[Dict]:
    """
    Targeted Data Augmentation (TA).

    Strategy:
      1. Count per-label frequency across training data.
      2. Identify minority labels (below `minority_threshold_pct` of max freq).
      3. For records containing ONLY majority-class entities:
         generate `augment_ratio` × N augmented copies.
      4. For records containing at least one minority-class entity:
         generate `minority_boost` augmented copies each.
      This rebalances the label distribution and increases format diversity.

    Args:
        train_data: Original training records.
        augment_ratio: Fraction of majority-class records to augment.
        minority_boost: How many augmented copies per minority-bearing record.
        minority_threshold_pct: Labels with freq < this % of max are "minority".
        aug_prob_per_span: Probability of transforming each individual span.
        seed: Random seed for reproducibility.

    Returns:
        Combined list: original + augmented records.
    """
    random.seed(seed)

    # --- 1. Label frequency analysis ---
    label_counts = Counter()
    for rec in train_data:
        for m in rec.get("privacy_mask", []):
            label_counts[m.get("label", "").strip().upper()] += 1

    if not label_counts:
        logger.warning("No PII labels found in training data. Skipping augmentation.")
        return train_data

    max_freq = max(label_counts.values())
    minority_threshold = max_freq * minority_threshold_pct
    minority_labels = {l for l, c in label_counts.items() if c < minority_threshold}

    logger.info(f"Label distribution: {dict(label_counts.most_common())}")
    if minority_labels:
        logger.info(f"Minority labels (threshold < {minority_threshold:.0f}): {sorted(minority_labels)}")
    else:
        logger.info("No extreme minority labels detected; applying uniform augmentation.")

    # --- 2. Split records into minority-bearing vs majority-only ---
    minority_records = []
    majority_records = []
    for rec in train_data:
        rec_labels = {m.get("label", "").strip().upper() for m in rec.get("privacy_mask", [])}
        if rec_labels & minority_labels:
            minority_records.append(rec)
        else:
            majority_records.append(rec)

    logger.info(f"Records with minority labels: {len(minority_records)}, majority-only: {len(majority_records)}")

    # --- 3. Generate augmented copies ---
    augmented = []

    # Minority: generate multiple copies
    for rec in minority_records:
        for _ in range(minority_boost):
            aug_rec = _augment_single_record(rec, aug_prob=aug_prob_per_span)
            if aug_rec:
                augmented.append(aug_rec)

    # Majority: sample a fraction and augment once
    n_majority_aug = int(len(majority_records) * augment_ratio)
    sampled_majority = random.sample(majority_records, min(n_majority_aug, len(majority_records)))
    for rec in sampled_majority:
        aug_rec = _augment_single_record(rec, aug_prob=aug_prob_per_span)
        if aug_rec:
            augmented.append(aug_rec)

    combined = train_data + augmented
    random.shuffle(combined)

    n_new = len(augmented)
    logger.info(
        f"Targeted Augmentation complete: {len(train_data)} original + "
        f"{n_new} augmented = {len(combined)} total"
    )
    return combined

// Device-compatible Montgomery arithmetic for the CUDA transcript kernel.

/// Number of 32-bit radix words in a BN254 scalar.
const FIELD_WORDS: usize = 8;

/// High-bit tag that keeps radix words in 64-bit device registers.
const WORD_TAG: u64 = 0x8000_0000_0000_0000;

/// Negative BN254 modulus inverse modulo `2^32` for Montgomery reduction.
const MONTGOMERY_MODULUS_INVERSE_WORD: u32 = 0xefff_ffff;

/// Least-significant limb of `R^2 mod p`, with `R = 2^256`.
const MONTGOMERY_R2_0: u64 = 0x1bb8_e645_ae21_6da7;

/// Second limb of `R^2 mod p`.
const MONTGOMERY_R2_1: u64 = 0x53fe_3ab1_e35c_59e3;

/// Third limb of `R^2 mod p`.
const MONTGOMERY_R2_2: u64 = 0x8c49_833d_53bb_8085;

/// Most-significant limb of `R^2 mod p`.
const MONTGOMERY_R2_3: u64 = 0x0216_d0b1_7f4e_44a5;

/// A BN254 scalar represented as eight tagged radix-2^32 words.
type FieldWords = [u64; 8];

/// Returns `R^2 mod p` for the four-limb Montgomery encoding.
fn device_montgomery_r2() -> FieldLimbs {
    [
        MONTGOMERY_R2_0,
        MONTGOMERY_R2_1,
        MONTGOMERY_R2_2,
        MONTGOMERY_R2_3,
    ]
}

/// Adds one carry word to the upper edge of a seventeen-word accumulator.
///
/// The explicit arms are intentional: cuda-oxide v0.2.0 does not yet lower
/// an indexed assignment through a mutable fixed-size slice.
fn add_word(mut accumulator: [u64; 17], index: usize, value: u64) -> [u64; 17] {
    macro_rules! add_carry {
        ($slot:literal, $carry:ident) => {
            if $carry != 0 {
                let word = (accumulator[$slot] ^ WORD_TAG) & 0xffff_ffff_u64;
                let sum = word + $carry;
                accumulator[$slot] = (sum & 0xffff_ffff_u64) | WORD_TAG;
                $carry = sum >> 32;
            }
        };
    }
    macro_rules! add_final_carry {
        ($slot:literal, $carry:ident) => {
            if $carry != 0 {
                let word = (accumulator[$slot] ^ WORD_TAG) & 0xffff_ffff_u64;
                let sum = word + $carry;
                accumulator[$slot] = (sum & 0xffff_ffff_u64) | WORD_TAG;
            }
        };
    }
    macro_rules! add_chain {
        ($($slot:literal),+; $last:literal) => {
            let mut carry = value;
            $(add_carry!($slot, carry);)+
            add_final_carry!($last, carry);
        };
    }
    match index {
        8 => { add_chain!(8, 9, 10, 11, 12, 13, 14, 15; 16); }
        9 => { add_chain!(9, 10, 11, 12, 13, 14, 15; 16); }
        10 => { add_chain!(10, 11, 12, 13, 14, 15; 16); }
        11 => { add_chain!(11, 12, 13, 14, 15; 16); }
        12 => { add_chain!(12, 13, 14, 15; 16); }
        13 => { add_chain!(13, 14, 15; 16); }
        14 => { add_chain!(14, 15; 16); }
        15 => { add_chain!(15; 16); }
        _ => {}
    }
    accumulator
}

/// Computes one Montgomery product with an eight-word CIOS reduction.
///
/// A 32-bit radix keeps each product-plus-carry below `2^64`, so the device
/// compiler can lower every inner step to ordinary exact `u64` arithmetic
/// without a software 128-bit helper.
pub(super) fn montgomery_mul(left: FieldLimbs, right: FieldLimbs) -> FieldLimbs {
    let modulus = device_field_modulus();
    let left_words = split_words(left);
    let right_words = split_words(right);
    let modulus_words = split_words(modulus);
    let mut product = [WORD_TAG; 17];
    let mut outer = 0_usize;
    while outer < FIELD_WORDS {
        let mut carry = 0_u64;
        let mut inner = 0_usize;
        while inner < FIELD_WORDS {
            let position = outer + inner;
            let left_word = (left_words[inner] ^ WORD_TAG) & 0xffff_ffff_u64;
            let right_word = (right_words[outer] ^ WORD_TAG) & 0xffff_ffff_u64;
            let product_word = (product[position] ^ WORD_TAG) & 0xffff_ffff_u64;
            let value = left_word * right_word + product_word + carry;
            product[position] = (value & 0xffff_ffff_u64) | WORD_TAG;
            carry = value >> 32;
            inner = inner.saturating_add(1);
        }
        product = add_word(product, outer + FIELD_WORDS, carry);

        let reduction = low_word(
            ((product[outer] ^ WORD_TAG) & 0xffff_ffff_u64)
                * u64::from(MONTGOMERY_MODULUS_INVERSE_WORD),
        );
        carry = 0;
        inner = 0;
        while inner < FIELD_WORDS {
            let position = outer + inner;
            let modulus_word = (modulus_words[inner] ^ WORD_TAG) & 0xffff_ffff_u64;
            let product_word = (product[position] ^ WORD_TAG) & 0xffff_ffff_u64;
            let value = modulus_word * u64::from(reduction) + product_word + carry;
            product[position] = (value & 0xffff_ffff_u64) | WORD_TAG;
            carry = value >> 32;
            inner = inner.saturating_add(1);
        }
        product = add_word(product, outer + FIELD_WORDS, carry);
        outer = outer.saturating_add(1);
    }
    let result = join_words([
        product[8],
        product[9],
        product[10],
        product[11],
        product[12],
        product[13],
        product[14],
        product[15],
    ]);
    if ((product[16] ^ WORD_TAG) & 0xffff_ffff_u64) != 0 || !field_less_than(result, modulus) {
        subtract_modulus(result, modulus)
    } else {
        result
    }
}

/// Splits four little-endian 64-bit limbs into eight tagged radix words.
fn split_words(value: FieldLimbs) -> FieldWords {
    let mut words = [0_u64; FIELD_WORDS];
    let mut limb = 0_usize;
    while limb < FIELD_LIMBS {
        let lower = value[limb] & u64::from(u32::MAX);
        let upper = value[limb] >> 32;
        words[limb * 2] = lower | WORD_TAG;
        words[limb * 2 + 1] = (upper & 0xffff_ffff_u64) | WORD_TAG;
        limb = limb.saturating_add(1);
    }
    words
}

/// Joins eight tagged radix words into four little-endian 64-bit limbs.
fn join_words(words: FieldWords) -> FieldLimbs {
    let mut limbs = [0_u64; FIELD_LIMBS];
    let mut limb = 0_usize;
    while limb < FIELD_LIMBS {
        let lower = (words[limb * 2] ^ WORD_TAG) & 0xffff_ffff_u64;
        let upper = (words[limb * 2 + 1] ^ WORD_TAG) & 0xffff_ffff_u64;
        limbs[limb] = lower | (upper << 32);
        limb = limb.saturating_add(1);
    }
    limbs
}

/// Truncates a masked 64-bit value to one exact 32-bit word.
fn low_word(value: u64) -> u32 {
    (value & u64::from(u32::MAX)) as u32
}

/// Converts an ordinary canonical field element into Montgomery form.
fn to_montgomery(value: FieldLimbs) -> FieldLimbs {
    montgomery_mul(value, device_montgomery_r2())
}

/// Converts a Montgomery field element back to ordinary canonical form.
fn from_montgomery(value: FieldLimbs) -> FieldLimbs {
    montgomery_mul(value, [1_u64, 0, 0, 0])
}

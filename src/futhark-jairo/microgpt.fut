--------- Generic Combinators ---------

def imap 'a : (n: i64) -> (i64 -> a) -> [n]a =
  \n f -> map f (iota n)

def imap1 = imap

def imap2 'a : (m: i64) -> (n: i64) -> (i64 -> i64 -> a) -> [m][n]a =
  \m n f -> imap m (\i -> imap n (f i))

def imap3 'a : (m: i64)
-> (n: i64)
-> (k: i64)
-> (i64 -> i64 -> i64 -> a) -> [m][n][k]a =
  \m n k f -> imap m (\i -> imap2 n k (f i))

def imap4 'a : (m: i64)
-> (n: i64)
-> (k: i64)
-> (l: i64)
-> (i64 -> i64 -> i64 -> i64 -> a) -> [m][n][k][l]a =
  \m n k l f -> imap m (\i -> imap3 n k l (f i))

def imap5 'a : (m: i64)
-> (n: i64)
-> (k: i64)
-> (l: i64)
-> (t: i64)
-> (i64 -> i64 -> i64 -> i64 -> i64 -> a) -> [m][n][k][l][t]a =
  \m n k l t f -> imap m (\i -> imap4 n k l t (f i))

def unzip7 [n] 'a 'b 'c 'd 'e 'f 'g : (a: [n](a, b, c, d, e, f, g)) -> ([n]a, [n]b, [n]c, [n]d, [n]e, [n]f, [n]g) =
  \a ->
    ( imap n (\i -> a[i].0)
    , imap n (\i -> a[i].1)
    , imap n (\i -> a[i].2)
    , imap n (\i -> a[i].3)
    , imap n (\i -> a[i].4)
    , imap n (\i -> a[i].5)
    , imap n (\i -> a[i].6)
    )

--==== Convolution Module ====--
module micrgpt (F: real) = {
  type real = F.t

  def fromi64 (n: i64) = F.from_fraction n 1 -- why from fraction?
  def zero = fromi64 0
  def one = fromi64 1

  def isum1 : (m: i64) -> (i64 -> real) -> real =
    \m f -> loop r = zero for i < m do r F.+ f i

  def isum2 : (m: i64)
  -> (n: i64)
  -> (i64 -> i64 -> real) -> real =
    \m n f -> loop r = zero for i < m do r F.+ isum1 n (f i)

  def isum3 : (m: i64)
  -> (n: i64)
  -> (k: i64)
  -> (i64 -> i64 -> i64 -> real) -> real =
    \n m k f -> loop r = zero for i < n do r F.+ isum2 m k (f i)

  def isum4 : (m: i64)
  -> (n: i64)
  -> (k: i64)
  -> (l: i64)
  -> (i64 -> i64 -> i64 -> i64 -> real) -> real =
    \n m k l f -> loop r = zero for i < n do r F.+ isum3 m k l (f i)

  def isum5 : (m: i64)
  -> (n: i64)
  -> (k: i64)
  -> (l: i64)
  -> (t: i64)
  -> (i64 -> i64 -> i64 -> i64 -> i64 -> real) -> real =
    \n m k l t f -> loop r = zero for i < n do r F.+ isum4 m k l t (f i)

  def sum (a: []real) : real =
    reduce (F.+) zero a

  --==== 2d cases ====--
  def sum2d (a: [][]real) : real =
    sum (map sum a)

  --==== Logistics ====--
  def logistics : real -> real =
    \e -> one F./ (one F.+ F.exp (F.neg e))

  --==== This is the generated function. ====--
  
      let x0 = (imap3 4 4 16 (\ x2_0 x2_1 x2_2 -> (inp[x2_0][x2_1][x2_2] F.* (one F./ (F.sqrt ((isum3 4 4 16 (\ x1_0 x1_1 x1_2 -> (inp[x2_0][x2_1][x2_2] F.* inp[x2_0][x2_1][x2_2]))) F./ fromi64 256))))))
      let x3 = (imap3 4 4 16 (\ x4_0 x4_1 x4_2 -> (isum3 4 4 16 (\ x5_0 x5_1 x5_2 -> (wq[x4_0][x4_1][x4_2][x5_0][x5_1][x5_2] F.* x0[x5_0][x5_1][x5_2])))))
      let x6 = (imap3 4 4 16 (\ x7_0 x7_1 x7_2 -> (isum3 4 4 16 (\ x8_0 x8_1 x8_2 -> (wk[x7_0][x7_1][x7_2][x8_0][x8_1][x8_2] F.* x0[x8_0][x8_1][x8_2])))))
      let x9 = (imap3 4 4 16 (\ x10_0 x10_1 x10_2 -> (isum3 4 4 16 (\ x11_0 x11_1 x11_2 -> (wv[x10_0][x10_1][x10_2][x11_0][x11_1][x11_2] F.* x0[x11_0][x11_1][x11_2])))))
      let x12 = (imap3 4 4 16 (\ x13_0 x13_1 x13_2 -> (isum1 4 (\ x14_0 -> (((F.exp ((isum1 16 (\ x16_0 -> (x3[x13_0][x13_1][x16_0] F.* x6[x13_0][x14_0][x16_0]))) F./ fromi64 2)) F.* (one F./ (isum2 4 4 (\ x15_0 x15_1 -> (F.exp ((isum1 16 (\ x17_0 -> (x3[x13_0][x15_0][x17_0] F.* x6[x13_0][x15_1][x17_0]))) F./ fromi64 2)))))) F.* x9[x13_0][x14_0][x13_2])))))
      let x18 = (imap3 4 4 16 (\ x19_0 x19_1 x19_2 -> (isum3 4 4 16 (\ x20_0 x20_1 x20_2 -> (wo[x19_0][x19_1][x19_2][x20_0][x20_1][x20_2] F.* x12[x20_0][x20_1][x20_2])))))
      let x21 = (imap3 4 4 16 (\ x22_0 x22_1 x22_2 -> (x18[x22_0][x22_1][x22_2] F.+ inp[x22_0][x22_1][x22_2])))
      let x23 = (imap3 4 4 16 (\ x25_0 x25_1 x25_2 -> (x21[x25_0][x25_1][x25_2] F.* (one F./ (F.sqrt ((isum3 4 4 16 (\ x24_0 x24_1 x24_2 -> (x21[x25_0][x25_1][x25_2] F.* x21[x25_0][x25_1][x25_2]))) F./ fromi64 256))))))
      let x26 = (imap4 4 4 4 16 (\ x27_0 x27_1 x27_2 x27_3 -> (isum3 4 4 16 (\ x28_0 x28_1 x28_2 -> (wf1[x27_0][x27_1][x27_2][x27_3][x28_0][x28_1][x28_2] F.* x23[x28_0][x28_1][x28_2])))))
      let x29 = (imap4 4 4 4 16 (\ x30_0 x30_1 x30_2 x30_3 -> (if (zero <= x26[x30_0][x30_1][x30_2][x30_3]) then x26[x30_0][x30_1][x30_2][x30_3] else zero)))
      let x31 = (imap3 4 4 16 (\ x32_0 x32_1 x32_2 -> (isum4 4 4 4 16 (\ x33_0 x33_1 x33_2 x33_3 -> (wf2[x32_0][x32_1][x32_2][x33_0][x33_1][x33_2][x33_3] F.* x29[x33_0][x33_1][x33_2][x33_3])))))
      let x34 = (imap3 4 4 16 (\ x35_0 x35_1 x35_2 -> (x31[x35_0][x35_1][x35_2] F.+ x21[x35_0][x35_1][x35_2])))
      let x36 = (imap3 4 4 16 (\ x37_0 x37_1 x37_2 -> one))
      let x38 = (imap3 4 4 16 (\ x39_0 x39_1 x39_2 -> x36[x39_0][x39_1][x39_2]))
      let x40 = (imap4 4 4 4 16 (\ x41_0 x41_1 x41_2 x41_3 -> (isum3 4 4 16 (\ x42_0 x42_1 x42_2 -> (x38[x42_0][x42_1][x42_2] F.* wf2[x42_0][x42_1][x42_2][x41_0][x41_1][x41_2][x41_3])))))
      let x43 = (imap4 4 4 4 16 (\ x44_0 x44_1 x44_2 x44_3 -> ((if (zero <= x26[x44_0][x44_1][x44_2][x44_3]) then one else zero) F.* x40[x44_0][x44_1][x44_2][x44_3])))
      let x45 = (imap3 4 4 16 (\ x46_0 x46_1 x46_2 -> (isum4 4 4 4 16 (\ x47_0 x47_1 x47_2 x47_3 -> (x43[x47_0][x47_1][x47_2][x47_3] F.* wf1[x47_0][x47_1][x47_2][x47_3][x46_0][x46_1][x46_2])))))
      let x48 = (imap3 4 4 16 (\ x57_0 x57_1 x57_2 -> ((x38[x57_0][x57_1][x57_2] F.+ (isum3 4 4 16 (\ x49_0 x49_1 x49_2 -> (((((F.neg ((x45[x57_0][x57_1][x57_2] F.* x21[x57_0][x57_1][x57_2]) F.* (one F./ ((F.sqrt ((isum3 4 4 16 (\ x50_0 x50_1 x50_2 -> (x21[x57_0][x57_1][x57_2] F.* x21[x57_0][x57_1][x57_2]))) F./ fromi64 256)) F.* (F.sqrt ((isum3 4 4 16 (\ x51_0 x51_1 x51_2 -> (x21[x57_0][x57_1][x57_2] F.* x21[x57_0][x57_1][x57_2]))) F./ fromi64 256)))))) F.* (one F./ ((x36[x57_0][x57_1][x57_2] F.+ x36[x57_0][x57_1][x57_2]) F.* (F.sqrt ((isum3 4 4 16 (\ x52_0 x52_1 x52_2 -> (x21[x57_0][x57_1][x57_2] F.* x21[x57_0][x57_1][x57_2]))) F./ fromi64 256))))) F./ fromi64 256) F.* x21[x57_0][x57_1][x57_2]) F.+ ((((F.neg ((x45[x57_0][x57_1][x57_2] F.* x21[x57_0][x57_1][x57_2]) F.* (one F./ ((F.sqrt ((isum3 4 4 16 (\ x53_0 x53_1 x53_2 -> (x21[x57_0][x57_1][x57_2] F.* x21[x57_0][x57_1][x57_2]))) F./ fromi64 256)) F.* (F.sqrt ((isum3 4 4 16 (\ x54_0 x54_1 x54_2 -> (x21[x57_0][x57_1][x57_2] F.* x21[x57_0][x57_1][x57_2]))) F./ fromi64 256)))))) F.* (one F./ ((x36[x57_0][x57_1][x57_2] F.+ x36[x57_0][x57_1][x57_2]) F.* (F.sqrt ((isum3 4 4 16 (\ x55_0 x55_1 x55_2 -> (x21[x57_0][x57_1][x57_2] F.* x21[x57_0][x57_1][x57_2]))) F./ fromi64 256))))) F./ fromi64 256) F.* x21[x57_0][x57_1][x57_2]))))) F.+ (x45[x57_0][x57_1][x57_2] F.* (one F./ (F.sqrt ((isum3 4 4 16 (\ x56_0 x56_1 x56_2 -> (x21[x57_0][x57_1][x57_2] F.* x21[x57_0][x57_1][x57_2]))) F./ fromi64 256)))))))
      let x58 = (imap3 4 4 16 (\ x59_0 x59_1 x59_2 -> x48[x59_0][x59_1][x59_2]))
      let x60 = (imap3 4 4 16 (\ x61_0 x61_1 x61_2 -> (isum3 4 4 16 (\ x62_0 x62_1 x62_2 -> (x58[x62_0][x62_1][x62_2] F.* wo[x62_0][x62_1][x62_2][x61_0][x61_1][x61_2])))))
      let x63 = (imap3 4 4 16 (\ x64_0 x64_1 x64_2 -> (isum1 4 (\ x65_0 -> (isum1 4 (\ x66_0 -> ((if ((x66_0 == x65_0)) then x60[x64_0][x65_0][x64_2] else zero) F.* ((F.exp ((isum1 16 (\ x68_0 -> (x3[x64_0][x66_0][x68_0] F.* x6[x64_0][x64_1][x68_0]))) F./ fromi64 2)) F.* (one F./ (isum2 4 4 (\ x67_0 x67_1 -> (F.exp ((isum1 16 (\ x69_0 -> (x3[x64_0][x67_0][x69_0] F.* x6[x64_0][x67_1][x69_0]))) F./ fromi64 2)))))))))))))
      let x70 = (imap3 4 4 16 (\ x77_0 x77_1 x77_2 -> (isum1 4 (\ x71_0 -> (isum1 4 (\ x72_0 -> (isum1 16 (\ x73_0 -> (isum1 4 (\ x74_0 -> (isum1 4 (\ x75_0 -> (isum2 4 4 (\ x76_0 x76_1 -> ((if ((x77_0 == x71_0)) then (isum2 4 4 (\ x78_0 x78_1 -> (isum1 4 (\ x79_0 -> (isum1 4 (\ x80_0 -> ((if ((x80_0 == x79_0)) then ((if ((x79_0 == x78_0) && (x77_1 == x78_1)) then ((F.exp ((isum1 16 (\ x83_0 -> (x3[x71_0][x78_0][x83_0] F.* x6[x71_0][x78_1][x83_0]))) F./ fromi64 2)) F.* (F.neg (((if ((x76_0 == x74_0)) then (if ((x76_1 == x75_0)) then ((if ((x74_0 == x72_0)) then x60[x71_0][x72_0][x73_0] else zero) F.* x9[x71_0][x75_0][x73_0]) else zero) else zero) F.* (F.exp ((isum1 16 (\ x84_0 -> (x3[x71_0][x76_0][x84_0] F.* x6[x71_0][x76_1][x84_0]))) F./ fromi64 2))) F.* (one F./ ((isum2 4 4 (\ x81_0 x81_1 -> (F.exp ((isum1 16 (\ x85_0 -> (x3[x71_0][x81_0][x85_0] F.* x6[x71_0][x81_1][x85_0]))) F./ fromi64 2)))) F.* (isum2 4 4 (\ x82_0 x82_1 -> (F.exp ((isum1 16 (\ x86_0 -> (x3[x71_0][x82_0][x86_0] F.* x6[x71_0][x82_1][x86_0]))) F./ fromi64 2))))))))) else zero) F./ fromi64 2) else zero) F.* x3[x71_0][x80_0][x77_2]))))))) else zero) F.+ (if ((x77_0 == x71_0)) then (isum1 4 (\ x87_0 -> (isum1 4 (\ x88_0 -> ((if ((x88_0 == x87_0)) then ((if ((x87_0 == x76_0) && (x77_1 == x76_1)) then ((F.exp ((isum1 16 (\ x90_0 -> (x3[x71_0][x76_0][x90_0] F.* x6[x71_0][x76_1][x90_0]))) F./ fromi64 2)) F.* ((if ((x76_0 == x74_0)) then (if ((x76_1 == x75_0)) then ((if ((x74_0 == x72_0)) then x60[x71_0][x72_0][x73_0] else zero) F.* x9[x71_0][x75_0][x73_0]) else zero) else zero) F.* (one F./ (isum2 4 4 (\ x89_0 x89_1 -> (F.exp ((isum1 16 (\ x91_0 -> (x3[x71_0][x89_0][x91_0] F.* x6[x71_0][x89_1][x91_0]))) F./ fromi64 2))))))) else zero) F./ fromi64 2) else zero) F.* x3[x71_0][x88_0][x77_2]))))) else zero))))))))))))))))
      let x92 = (imap3 4 4 16 (\ x99_0 x99_1 x99_2 -> (isum1 4 (\ x93_0 -> (isum1 4 (\ x94_0 -> (isum1 16 (\ x95_0 -> (isum1 4 (\ x96_0 -> (isum1 4 (\ x97_0 -> (isum2 4 4 (\ x98_0 x98_1 -> ((if ((x99_0 == x93_0)) then (isum2 4 4 (\ x100_0 x100_1 -> (isum1 4 (\ x101_0 -> (isum1 4 (\ x102_0 -> ((if ((x99_1 == x101_0)) then ((if ((x101_0 == x100_0) && (x102_0 == x100_1)) then ((F.exp ((isum1 16 (\ x105_0 -> (x3[x93_0][x100_0][x105_0] F.* x6[x93_0][x100_1][x105_0]))) F./ fromi64 2)) F.* (F.neg (((if ((x98_0 == x96_0)) then (if ((x98_1 == x97_0)) then ((if ((x96_0 == x94_0)) then x60[x93_0][x94_0][x95_0] else zero) F.* x9[x93_0][x97_0][x95_0]) else zero) else zero) F.* (F.exp ((isum1 16 (\ x106_0 -> (x3[x93_0][x98_0][x106_0] F.* x6[x93_0][x98_1][x106_0]))) F./ fromi64 2))) F.* (one F./ ((isum2 4 4 (\ x103_0 x103_1 -> (F.exp ((isum1 16 (\ x107_0 -> (x3[x93_0][x103_0][x107_0] F.* x6[x93_0][x103_1][x107_0]))) F./ fromi64 2)))) F.* (isum2 4 4 (\ x104_0 x104_1 -> (F.exp ((isum1 16 (\ x108_0 -> (x3[x93_0][x104_0][x108_0] F.* x6[x93_0][x104_1][x108_0]))) F./ fromi64 2))))))))) else zero) F./ fromi64 2) else zero) F.* x6[x93_0][x102_0][x99_2]))))))) else zero) F.+ (if ((x99_0 == x93_0)) then (isum1 4 (\ x109_0 -> (isum1 4 (\ x110_0 -> ((if ((x99_1 == x109_0)) then ((if ((x109_0 == x98_0) && (x110_0 == x98_1)) then ((F.exp ((isum1 16 (\ x112_0 -> (x3[x93_0][x98_0][x112_0] F.* x6[x93_0][x98_1][x112_0]))) F./ fromi64 2)) F.* ((if ((x98_0 == x96_0)) then (if ((x98_1 == x97_0)) then ((if ((x96_0 == x94_0)) then x60[x93_0][x94_0][x95_0] else zero) F.* x9[x93_0][x97_0][x95_0]) else zero) else zero) F.* (one F./ (isum2 4 4 (\ x111_0 x111_1 -> (F.exp ((isum1 16 (\ x113_0 -> (x3[x93_0][x111_0][x113_0] F.* x6[x93_0][x111_1][x113_0]))) F./ fromi64 2))))))) else zero) F./ fromi64 2) else zero) F.* x6[x93_0][x110_0][x99_2]))))) else zero))))))))))))))))
      let x114 = (imap3 4 4 16 (\ x115_0 x115_1 x115_2 -> (((isum3 4 4 16 (\ x116_0 x116_1 x116_2 -> (x63[x116_0][x116_1][x116_2] F.* wv[x116_0][x116_1][x116_2][x115_0][x115_1][x115_2]))) F.+ (isum3 4 4 16 (\ x117_0 x117_1 x117_2 -> (x70[x117_0][x117_1][x117_2] F.* wk[x117_0][x117_1][x117_2][x115_0][x115_1][x115_2])))) F.+ (isum3 4 4 16 (\ x118_0 x118_1 x118_2 -> (x92[x118_0][x118_1][x118_2] F.* wq[x118_0][x118_1][x118_2][x115_0][x115_1][x115_2]))))))

      let dinp = (imap3 4 4 16 (\ x127_0 x127_1 x127_2 -> ((x58[x127_0][x127_1][x127_2] F.+ (isum3 4 4 16 (\ x119_0 x119_1 x119_2 -> (((((F.neg ((x114[x127_0][x127_1][x127_2] F.* inp[x127_0][x127_1][x127_2]) F.* (one F./ ((F.sqrt ((isum3 4 4 16 (\ x120_0 x120_1 x120_2 -> (inp[x127_0][x127_1][x127_2] F.* inp[x127_0][x127_1][x127_2]))) F./ fromi64 256)) F.* (F.sqrt ((isum3 4 4 16 (\ x121_0 x121_1 x121_2 -> (inp[x127_0][x127_1][x127_2] F.* inp[x127_0][x127_1][x127_2]))) F./ fromi64 256)))))) F.* (one F./ ((x36[x127_0][x127_1][x127_2] F.+ x36[x127_0][x127_1][x127_2]) F.* (F.sqrt ((isum3 4 4 16 (\ x122_0 x122_1 x122_2 -> (inp[x127_0][x127_1][x127_2] F.* inp[x127_0][x127_1][x127_2]))) F./ fromi64 256))))) F./ fromi64 256) F.* inp[x127_0][x127_1][x127_2]) F.+ ((((F.neg ((x114[x127_0][x127_1][x127_2] F.* inp[x127_0][x127_1][x127_2]) F.* (one F./ ((F.sqrt ((isum3 4 4 16 (\ x123_0 x123_1 x123_2 -> (inp[x127_0][x127_1][x127_2] F.* inp[x127_0][x127_1][x127_2]))) F./ fromi64 256)) F.* (F.sqrt ((isum3 4 4 16 (\ x124_0 x124_1 x124_2 -> (inp[x127_0][x127_1][x127_2] F.* inp[x127_0][x127_1][x127_2]))) F./ fromi64 256)))))) F.* (one F./ ((x36[x127_0][x127_1][x127_2] F.+ x36[x127_0][x127_1][x127_2]) F.* (F.sqrt ((isum3 4 4 16 (\ x125_0 x125_1 x125_2 -> (inp[x127_0][x127_1][x127_2] F.* inp[x127_0][x127_1][x127_2]))) F./ fromi64 256))))) F./ fromi64 256) F.* inp[x127_0][x127_1][x127_2]))))) F.+ (x114[x127_0][x127_1][x127_2] F.* (one F./ (F.sqrt ((isum3 4 4 16 (\ x126_0 x126_1 x126_2 -> (inp[x127_0][x127_1][x127_2] F.* inp[x127_0][x127_1][x127_2]))) F./ fromi64 256)))))))
      let dwq = (imap6 4 4 16 4 4 16 (\ x128_0 x128_1 x128_2 x128_3 x128_4 x128_5 -> (x92[x128_0][x128_1][x128_2] F.* x0[x128_3][x128_4][x128_5])))
      let dwk = (imap6 4 4 16 4 4 16 (\ x129_0 x129_1 x129_2 x129_3 x129_4 x129_5 -> (x70[x129_0][x129_1][x129_2] F.* x0[x129_3][x129_4][x129_5])))
      let dwv = (imap6 4 4 16 4 4 16 (\ x130_0 x130_1 x130_2 x130_3 x130_4 x130_5 -> (x63[x130_0][x130_1][x130_2] F.* x0[x130_3][x130_4][x130_5])))
      let dwo = (imap6 4 4 16 4 4 16 (\ x131_0 x131_1 x131_2 x131_3 x131_4 x131_5 -> (x58[x131_0][x131_1][x131_2] F.* x12[x131_3][x131_4][x131_5])))
      let dwf1 = (imap7 4 4 4 16 4 4 16 (\ x132_0 x132_1 x132_2 x132_3 x132_4 x132_5 x132_6 -> (x43[x132_0][x132_1][x132_2][x132_3] F.* x23[x132_4][x132_5][x132_6])))
      let dwf2 = (imap7 4 4 16 4 4 4 16 (\ x133_0 x133_1 x133_2 x133_3 x133_4 x133_5 x133_6 -> (x38[x133_0][x133_1][x133_2] F.* x29[x133_3][x133_4][x133_5][x133_6])))

  --==== Attention. ====--
  -- def attention :
  -- (queries: [4][16]real) -> (keys: [4][16]real) -> (values: [4][16]real) -> [4][16]real =
  --   \(queries: [4][16]real) (keys: [4][16]real) (values: [4][16]real) ->
  --     (imap2 4 16 (\ x0_0 x0_1 -> (isum1 4 (\ x1_0 -> (((F.exp ((isum1 16 (\ x3_0 -> (queries[x0_0][x3_0] F.* keys[x1_0][x3_0]))) F./ fromi64 2)) F.* (one F./ (isum2 4 4 (\ x2_0 x2_1 -> (F.exp ((isum1 16 (\ x4_0 -> (queries[x2_0][x4_0] F.* keys[x2_1][x4_0]))) F./ fromi64 2)))))) F.* values[x1_0][x0_1])))))

  --==== Attention Gradient. ====--
  -- inputs : queries keys values : ar (4 x 16)
  -- def train_attention :
  -- (queries: [4][16]real) -> (keys: [4][16]real) -> (values: [4][16]real) -> ([4][16]real , [4][16]real , [4][16]real) =
  -- \(queries: [4][16]real) (keys: [4][16]real) (values: [4][16]real) ->
  --     let dqueries = (imap2 4 16 (\ x5_0 x5_1 -> (isum1 4 (\ x0_0 -> (isum1 16 (\ x1_0 -> (isum1 4 (\ x2_0 -> (isum1 4 (\ x3_0 -> (isum2 4 4 (\ x4_0 x4_1 -> ((isum2 4 4 (\ x6_0 x6_1 -> (isum1 4 (\ x7_0 -> (isum1 4 (\ x8_0 -> ((if ((x5_0 == x7_0)) then ((if ((x7_0 == x6_0) && (x8_0 == x6_1)) then ((F.exp ((isum1 16 (\ x11_0 -> (queries[x6_0][x11_0] F.* keys[x6_1][x11_0]))) F./ fromi64 2)) F.* (F.neg (((if ((x4_0 == x2_0)) then (if ((x4_1 == x3_0)) then ((if ((x2_0 == x0_0)) then one else zero) F.* values[x3_0][x1_0]) else zero) else zero) F.* (F.exp ((isum1 16 (\ x12_0 -> (queries[x4_0][x12_0] F.* keys[x4_1][x12_0]))) F./ fromi64 2))) F.* (one F./ ((isum2 4 4 (\ x9_0 x9_1 -> (F.exp ((isum1 16 (\ x13_0 -> (queries[x9_0][x13_0] F.* keys[x9_1][x13_0]))) F./ fromi64 2)))) F.* (isum2 4 4 (\ x10_0 x10_1 -> (F.exp ((isum1 16 (\ x14_0 -> (queries[x10_0][x14_0] F.* keys[x10_1][x14_0]))) F./ fromi64 2))))))))) else zero) F./ fromi64 2) else zero) F.* keys[x8_0][x5_1]))))))) F.+ (isum1 4 (\ x15_0 -> (isum1 4 (\ x16_0 -> ((if ((x5_0 == x15_0)) then ((if ((x15_0 == x4_0) && (x16_0 == x4_1)) then ((F.exp ((isum1 16 (\ x18_0 -> (queries[x4_0][x18_0] F.* keys[x4_1][x18_0]))) F./ fromi64 2)) F.* ((if ((x4_0 == x2_0)) then (if ((x4_1 == x3_0)) then ((if ((x2_0 == x0_0)) then one else zero) F.* values[x3_0][x1_0]) else zero) else zero) F.* (one F./ (isum2 4 4 (\ x17_0 x17_1 -> (F.exp ((isum1 16 (\ x19_0 -> (queries[x17_0][x19_0] F.* keys[x17_1][x19_0]))) F./ fromi64 2))))))) else zero) F./ fromi64 2) else zero) F.* keys[x16_0][x5_1]))))))))))))))))))
  --     let dkeys = (imap2 4 16 (\ x25_0 x25_1 -> (isum1 4 (\ x20_0 -> (isum1 16 (\ x21_0 -> (isum1 4 (\ x22_0 -> (isum1 4 (\ x23_0 -> (isum2 4 4 (\ x24_0 x24_1 -> ((isum2 4 4 (\ x26_0 x26_1 -> (isum1 4 (\ x27_0 -> (isum1 4 (\ x28_0 -> ((if ((x28_0 == x27_0)) then ((if ((x27_0 == x26_0) && (x25_0 == x26_1)) then ((F.exp ((isum1 16 (\ x31_0 -> (queries[x26_0][x31_0] F.* keys[x26_1][x31_0]))) F./ fromi64 2)) F.* (F.neg (((if ((x24_0 == x22_0)) then (if ((x24_1 == x23_0)) then ((if ((x22_0 == x20_0)) then one else zero) F.* values[x23_0][x21_0]) else zero) else zero) F.* (F.exp ((isum1 16 (\ x32_0 -> (queries[x24_0][x32_0] F.* keys[x24_1][x32_0]))) F./ fromi64 2))) F.* (one F./ ((isum2 4 4 (\ x29_0 x29_1 -> (F.exp ((isum1 16 (\ x33_0 -> (queries[x29_0][x33_0] F.* keys[x29_1][x33_0]))) F./ fromi64 2)))) F.* (isum2 4 4 (\ x30_0 x30_1 -> (F.exp ((isum1 16 (\ x34_0 -> (queries[x30_0][x34_0] F.* keys[x30_1][x34_0]))) F./ fromi64 2))))))))) else zero) F./ fromi64 2) else zero) F.* queries[x28_0][x25_1]))))))) F.+ (isum1 4 (\ x35_0 -> (isum1 4 (\ x36_0 -> ((if ((x36_0 == x35_0)) then ((if ((x35_0 == x24_0) && (x25_0 == x24_1)) then ((F.exp ((isum1 16 (\ x38_0 -> (queries[x24_0][x38_0] F.* keys[x24_1][x38_0]))) F./ fromi64 2)) F.* ((if ((x24_0 == x22_0)) then (if ((x24_1 == x23_0)) then ((if ((x22_0 == x20_0)) then one else zero) F.* values[x23_0][x21_0]) else zero) else zero) F.* (one F./ (isum2 4 4 (\ x37_0 x37_1 -> (F.exp ((isum1 16 (\ x39_0 -> (queries[x37_0][x39_0] F.* keys[x37_1][x39_0]))) F./ fromi64 2))))))) else zero) F./ fromi64 2) else zero) F.* queries[x36_0][x25_1]))))))))))))))))))
  --     let dvalues = (imap2 4 16 (\ x40_0 x40_1 -> (isum1 4 (\ x41_0 -> (isum1 4 (\ x42_0 -> ((if ((x42_0 == x41_0)) then one else zero) F.* ((F.exp ((isum1 16 (\ x44_0 -> (queries[x42_0][x44_0] F.* keys[x40_0][x44_0]))) F./ fromi64 2)) F.* (one F./ (isum2 4 4 (\ x43_0 x43_1 -> (F.exp ((isum1 16 (\ x45_0 -> (queries[x43_0][x45_0] F.* keys[x43_1][x45_0]))) F./ fromi64 2)))))))))))))
  --     in (dqueries, dkeys, dvalues)
}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE EmptyCase #-}
{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# OPTIONS_GHC -Wno-overlapping-patterns #-}

module MAlonzo.Code.IO.Primitive.Handle where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.IO
import qualified MAlonzo.Code.Agda.Builtin.Maybe
import qualified MAlonzo.Code.Agda.Builtin.Unit

import System.IO
data AgdaBufferMode
  = AgdaNoBuffering
  | AgdaLineBuffering
  | AgdaBlockBuffering (Maybe Integer)
toBufferMode :: AgdaBufferMode -> BufferMode
toBufferMode x = case x of
  AgdaNoBuffering       -> NoBuffering
  AgdaLineBuffering     -> LineBuffering
  AgdaBlockBuffering mi -> BlockBuffering (fromIntegral <$> mi)
fromBufferMode :: BufferMode -> AgdaBufferMode
fromBufferMode x = case x of
  NoBuffering       -> AgdaNoBuffering
  LineBuffering     -> AgdaLineBuffering
  BlockBuffering mi -> AgdaBlockBuffering (fromIntegral <$> mi)
-- IO.Primitive.Handle.BufferMode
d_BufferMode_6 = ()
type T_BufferMode_6 = AgdaBufferMode
pattern C_noBuffering_8 = AgdaNoBuffering
pattern C_lineBuffering_10 = AgdaLineBuffering
pattern C_blockBuffering_12 a0 = AgdaBlockBuffering a0
check_noBuffering_8 :: T_BufferMode_6
check_noBuffering_8 = AgdaNoBuffering
check_lineBuffering_10 :: T_BufferMode_6
check_lineBuffering_10 = AgdaLineBuffering
check_blockBuffering_12 ::
  MAlonzo.Code.Agda.Builtin.Maybe.T_Maybe_10 () Integer ->
  T_BufferMode_6
check_blockBuffering_12 = AgdaBlockBuffering
cover_BufferMode_6 :: AgdaBufferMode -> ()
cover_BufferMode_6 x
  = case x of
      AgdaNoBuffering -> ()
      AgdaLineBuffering -> ()
      AgdaBlockBuffering _ -> ()
-- IO.Primitive.Handle.Handle
type T_Handle_14 = Handle
d_Handle_14
  = error
      "MAlonzo Runtime Error: postulate evaluated: IO.Primitive.Handle.Handle"
-- IO.Primitive.Handle.stdin
d_stdin_16 :: T_Handle_14
d_stdin_16 = stdin
-- IO.Primitive.Handle.stdout
d_stdout_18 :: T_Handle_14
d_stdout_18 = stdout
-- IO.Primitive.Handle.stderr
d_stderr_20 :: T_Handle_14
d_stderr_20 = stderr
-- IO.Primitive.Handle.hSetBuffering
d_hSetBuffering_22 ::
  T_Handle_14 ->
  T_BufferMode_6 ->
  MAlonzo.Code.Agda.Builtin.IO.T_IO_8
    () MAlonzo.Code.Agda.Builtin.Unit.T_'8868'_6
d_hSetBuffering_22 = \ h -> hSetBuffering h . toBufferMode
-- IO.Primitive.Handle.hGetBuffering
d_hGetBuffering_24 ::
  T_Handle_14 ->
  MAlonzo.Code.Agda.Builtin.IO.T_IO_8 () T_BufferMode_6
d_hGetBuffering_24 = fmap fromBufferMode . hGetBuffering
-- IO.Primitive.Handle.hFlush
d_hFlush_26 ::
  T_Handle_14 ->
  MAlonzo.Code.Agda.Builtin.IO.T_IO_8
    () MAlonzo.Code.Agda.Builtin.Unit.T_'8868'_6
d_hFlush_26 = hFlush

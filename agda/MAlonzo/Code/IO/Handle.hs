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

module MAlonzo.Code.IO.Handle where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Primitive
import qualified MAlonzo.Code.IO.Base
import qualified MAlonzo.Code.IO.Primitive.Handle

-- IO.Handle.hSetBuffering
d_hSetBuffering_6 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.IO.Primitive.Handle.T_Handle_14 ->
  MAlonzo.Code.IO.Primitive.Handle.T_BufferMode_6 ->
  MAlonzo.Code.IO.Base.T_IO_20
d_hSetBuffering_6 v0 v1 v2
  = coe
      MAlonzo.Code.IO.Base.d_lift'8242'_142 (coe v0)
      (coe MAlonzo.Code.IO.Primitive.Handle.d_hSetBuffering_22 v1 v2)
-- IO.Handle.hGetBuffering
d_hGetBuffering_12 ::
  MAlonzo.Code.IO.Primitive.Handle.T_Handle_14 ->
  MAlonzo.Code.IO.Base.T_IO_20
d_hGetBuffering_12 v0
  = coe
      MAlonzo.Code.IO.Base.C_lift_26
      (coe MAlonzo.Code.IO.Primitive.Handle.d_hGetBuffering_24 v0)
-- IO.Handle.hFlush
d_hFlush_16 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.IO.Primitive.Handle.T_Handle_14 ->
  MAlonzo.Code.IO.Base.T_IO_20
d_hFlush_16 v0 v1
  = coe
      MAlonzo.Code.IO.Base.d_lift'8242'_142 (coe v0)
      (coe MAlonzo.Code.IO.Primitive.Handle.d_hFlush_26 v1)

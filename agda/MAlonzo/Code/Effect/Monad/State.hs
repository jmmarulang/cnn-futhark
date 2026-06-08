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

module MAlonzo.Code.Effect.Monad.State where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.Sigma
import qualified MAlonzo.Code.Agda.Primitive
import qualified MAlonzo.Code.Effect.Applicative
import qualified MAlonzo.Code.Effect.Functor
import qualified MAlonzo.Code.Effect.Monad
import qualified MAlonzo.Code.Effect.Monad.Identity
import qualified MAlonzo.Code.Effect.Monad.State.Transformer
import qualified MAlonzo.Code.Effect.Monad.State.Transformer.Base

-- Effect.Monad.State.State
d_State_16 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 -> () -> () -> ()
d_State_16 = erased
-- Effect.Monad.State.runState
d_runState_20 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14
d_runState_20 ~v0 ~v1 ~v2 v3 v4 = du_runState_20 v3 v4
du_runState_20 ::
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58 ->
  AgdaAny -> MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14
du_runState_20 v0 v1
  = coe
      MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20
      (coe
         MAlonzo.Code.Effect.Monad.State.Transformer.Base.d_runStateT_68 v0
         v1)
-- Effect.Monad.State.evalState
d_evalState_26 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58 ->
  AgdaAny -> AgdaAny
d_evalState_26 ~v0 ~v1 ~v2 v3 v4 = du_evalState_26 v3 v4
du_evalState_26 ::
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58 ->
  AgdaAny -> AgdaAny
du_evalState_26 v0 v1
  = coe
      MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20
      (coe
         MAlonzo.Code.Effect.Monad.State.Transformer.Base.du_evalStateT_72
         (coe MAlonzo.Code.Effect.Monad.Identity.du_functor_24) (coe v0)
         (coe v1))
-- Effect.Monad.State.execState
d_execState_32 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58 ->
  AgdaAny -> AgdaAny
d_execState_32 ~v0 ~v1 ~v2 v3 v4 = du_execState_32 v3 v4
du_execState_32 ::
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58 ->
  AgdaAny -> AgdaAny
du_execState_32 v0 v1
  = coe
      MAlonzo.Code.Effect.Monad.Identity.d_runIdentity_20
      (coe
         MAlonzo.Code.Effect.Monad.State.Transformer.Base.du_execStateT_90
         (coe MAlonzo.Code.Effect.Monad.Identity.du_functor_24) (coe v0)
         (coe v1))
-- Effect.Monad.State.functor
d_functor_38 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> MAlonzo.Code.Effect.Functor.T_RawFunctor_24
d_functor_38 ~v0 ~v1 = du_functor_38
du_functor_38 :: MAlonzo.Code.Effect.Functor.T_RawFunctor_24
du_functor_38
  = coe
      MAlonzo.Code.Effect.Monad.State.Transformer.du_functor_24
      (coe MAlonzo.Code.Effect.Monad.Identity.du_functor_24)
-- Effect.Monad.State.applicative
d_applicative_40 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> MAlonzo.Code.Effect.Applicative.T_RawApplicative_20
d_applicative_40 ~v0 ~v1 = du_applicative_40
du_applicative_40 ::
  MAlonzo.Code.Effect.Applicative.T_RawApplicative_20
du_applicative_40
  = coe
      MAlonzo.Code.Effect.Monad.State.Transformer.du_applicative_48
      (coe MAlonzo.Code.Effect.Monad.Identity.du_monad_36)
-- Effect.Monad.State.monad
d_monad_42 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> MAlonzo.Code.Effect.Monad.T_RawMonad_24
d_monad_42 ~v0 ~v1 = du_monad_42
du_monad_42 :: MAlonzo.Code.Effect.Monad.T_RawMonad_24
du_monad_42
  = coe
      MAlonzo.Code.Effect.Monad.State.Transformer.du_monad_238
      (coe MAlonzo.Code.Effect.Monad.Identity.du_monad_36)
-- Effect.Monad.State.join
d_join_44 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58
d_join_44 ~v0 ~v1 ~v2 = du_join_44
du_join_44 ::
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58 ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_StateT_58
du_join_44
  = coe MAlonzo.Code.Effect.Monad.du_join_160 (coe du_monad_42)
-- Effect.Monad.State.monadState
d_monadState_46 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28
d_monadState_46 ~v0 ~v1 = du_monadState_46
du_monadState_46 ::
  MAlonzo.Code.Effect.Monad.State.Transformer.Base.T_RawMonadState_28
du_monadState_46
  = coe
      MAlonzo.Code.Effect.Monad.State.Transformer.du_monadState_460
      (coe MAlonzo.Code.Effect.Monad.Identity.du_monad_36)

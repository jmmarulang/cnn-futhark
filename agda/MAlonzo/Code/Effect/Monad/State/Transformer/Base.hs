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

module MAlonzo.Code.Effect.Monad.State.Transformer.Base where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.Sigma
import qualified MAlonzo.Code.Agda.Primitive
import qualified MAlonzo.Code.Effect.Functor
import qualified MAlonzo.Code.Function.Base

-- Effect.Monad.State.Transformer.Base.RawMonadState
d_RawMonadState_28 a0 a1 a2 a3 = ()
data T_RawMonadState_28
  = C_constructor_46 (() -> (AgdaAny -> AgdaAny) -> AgdaAny)
                     ((AgdaAny -> AgdaAny) -> AgdaAny)
-- Effect.Monad.State.Transformer.Base.RawMonadState.gets
d_gets_38 ::
  T_RawMonadState_28 -> () -> (AgdaAny -> AgdaAny) -> AgdaAny
d_gets_38 v0
  = case coe v0 of
      C_constructor_46 v1 v2 -> coe v1
      _ -> MAlonzo.RTE.mazUnreachableError
-- Effect.Monad.State.Transformer.Base.RawMonadState.modify
d_modify_40 ::
  T_RawMonadState_28 -> (AgdaAny -> AgdaAny) -> AgdaAny
d_modify_40 v0
  = case coe v0 of
      C_constructor_46 v1 v2 -> coe v2
      _ -> MAlonzo.RTE.mazUnreachableError
-- Effect.Monad.State.Transformer.Base.RawMonadState.put
d_put_42 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> (() -> ()) -> T_RawMonadState_28 -> AgdaAny -> AgdaAny
d_put_42 ~v0 ~v1 ~v2 ~v3 v4 = du_put_42 v4
du_put_42 :: T_RawMonadState_28 -> AgdaAny -> AgdaAny
du_put_42 v0
  = coe
      MAlonzo.Code.Function.Base.du__'8728''8242'__216
      (coe d_modify_40 (coe v0)) (coe (\ v1 v2 -> v1))
-- Effect.Monad.State.Transformer.Base.RawMonadState.get
d_get_44 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> (() -> ()) -> T_RawMonadState_28 -> AgdaAny
d_get_44 ~v0 ~v1 ~v2 ~v3 v4 = du_get_44 v4
du_get_44 :: T_RawMonadState_28 -> AgdaAny
du_get_44 v0 = coe d_gets_38 v0 erased (\ v1 -> v1)
-- Effect.Monad.State.Transformer.Base.StateT
d_StateT_58 a0 a1 a2 a3 a4 = ()
newtype T_StateT_58 = C_mkStateT_70 (AgdaAny -> AgdaAny)
-- Effect.Monad.State.Transformer.Base.StateT.runStateT
d_runStateT_68 :: T_StateT_58 -> AgdaAny -> AgdaAny
d_runStateT_68 v0
  = case coe v0 of
      C_mkStateT_70 v1 -> coe v1
      _ -> MAlonzo.RTE.mazUnreachableError
-- Effect.Monad.State.Transformer.Base.evalStateT
d_evalStateT_72 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  () ->
  MAlonzo.Code.Effect.Functor.T_RawFunctor_24 ->
  T_StateT_58 -> AgdaAny -> AgdaAny
d_evalStateT_72 ~v0 ~v1 ~v2 ~v3 ~v4 v5 v6 v7
  = du_evalStateT_72 v5 v6 v7
du_evalStateT_72 ::
  MAlonzo.Code.Effect.Functor.T_RawFunctor_24 ->
  T_StateT_58 -> AgdaAny -> AgdaAny
du_evalStateT_72 v0 v1 v2
  = coe
      MAlonzo.Code.Effect.Functor.d__'60''36''62'__30 v0 erased erased
      (\ v3 -> MAlonzo.Code.Agda.Builtin.Sigma.d_snd_30 (coe v3))
      (coe d_runStateT_68 v1 v2)
-- Effect.Monad.State.Transformer.Base.execStateT
d_execStateT_90 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (() -> ()) ->
  () ->
  () ->
  MAlonzo.Code.Effect.Functor.T_RawFunctor_24 ->
  T_StateT_58 -> AgdaAny -> AgdaAny
d_execStateT_90 ~v0 ~v1 ~v2 ~v3 ~v4 v5 v6 v7
  = du_execStateT_90 v5 v6 v7
du_execStateT_90 ::
  MAlonzo.Code.Effect.Functor.T_RawFunctor_24 ->
  T_StateT_58 -> AgdaAny -> AgdaAny
du_execStateT_90 v0 v1 v2
  = coe
      MAlonzo.Code.Effect.Functor.d__'60''36''62'__30 v0 erased erased
      (\ v3 -> MAlonzo.Code.Agda.Builtin.Sigma.d_fst_28 (coe v3))
      (coe d_runStateT_68 v1 v2)

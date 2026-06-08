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

module MAlonzo.Code.Effect.Monad.Identity where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Primitive
import qualified MAlonzo.Code.Effect.Applicative
import qualified MAlonzo.Code.Effect.Comonad
import qualified MAlonzo.Code.Effect.Functor
import qualified MAlonzo.Code.Effect.Monad
import qualified MAlonzo.Code.Function.Base

-- Effect.Monad.Identity.Identity
d_Identity_14 a0 a1 = ()
newtype T_Identity_14 = C_mkIdentity_22 AgdaAny
-- Effect.Monad.Identity.Identity.runIdentity
d_runIdentity_20 :: T_Identity_14 -> AgdaAny
d_runIdentity_20 v0
  = case coe v0 of
      C_mkIdentity_22 v1 -> coe v1
      _ -> MAlonzo.RTE.mazUnreachableError
-- Effect.Monad.Identity.functor
d_functor_24 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Effect.Functor.T_RawFunctor_24
d_functor_24 ~v0 = du_functor_24
du_functor_24 :: MAlonzo.Code.Effect.Functor.T_RawFunctor_24
du_functor_24
  = coe
      MAlonzo.Code.Effect.Functor.C_constructor_44
      (coe
         (\ v0 v1 v2 v3 ->
            coe C_mkIdentity_22 (coe v2 (d_runIdentity_20 (coe v3)))))
-- Effect.Monad.Identity.applicative
d_applicative_30 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Effect.Applicative.T_RawApplicative_20
d_applicative_30 ~v0 = du_applicative_30
du_applicative_30 ::
  MAlonzo.Code.Effect.Applicative.T_RawApplicative_20
du_applicative_30
  = coe
      MAlonzo.Code.Effect.Applicative.C_constructor_78
      (coe du_functor_24) (coe (\ v0 -> coe C_mkIdentity_22))
      (coe
         (\ v0 v1 v2 v3 ->
            coe
              C_mkIdentity_22
              (let v4 = d_runIdentity_20 (coe v2) in
               coe (coe v4 (d_runIdentity_20 (coe v3))))))
-- Effect.Monad.Identity.monad
d_monad_36 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Effect.Monad.T_RawMonad_24
d_monad_36 ~v0 = du_monad_36
du_monad_36 :: MAlonzo.Code.Effect.Monad.T_RawMonad_24
du_monad_36
  = coe
      MAlonzo.Code.Effect.Monad.C_constructor_98 (coe du_applicative_30)
      (coe
         (\ v0 v1 ->
            coe
              MAlonzo.Code.Function.Base.du__'8728''8242'__216
              (coe MAlonzo.Code.Function.Base.du__'124''62''8242'__232)
              (coe (\ v2 -> d_runIdentity_20 (coe v2)))))
-- Effect.Monad.Identity.comonad
d_comonad_38 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Effect.Comonad.T_RawComonad_22
d_comonad_38 ~v0 = du_comonad_38
du_comonad_38 :: MAlonzo.Code.Effect.Comonad.T_RawComonad_22
du_comonad_38
  = coe
      MAlonzo.Code.Effect.Comonad.C_constructor_52
      (coe (\ v0 v1 -> d_runIdentity_20 (coe v1)))
      (coe (\ v0 v1 v2 v3 -> coe C_mkIdentity_22 (coe v2 v3)))
-- Effect.Monad.Identity.join
d_join_44 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> T_Identity_14 -> T_Identity_14
d_join_44 ~v0 ~v1 = du_join_44
du_join_44 :: T_Identity_14 -> T_Identity_14
du_join_44
  = coe MAlonzo.Code.Effect.Monad.du_join_160 (coe du_monad_36)

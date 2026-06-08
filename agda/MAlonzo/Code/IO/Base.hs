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

module MAlonzo.Code.IO.Base where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.Coinduction
import qualified MAlonzo.Code.Agda.Builtin.IO
import qualified MAlonzo.Code.Agda.Builtin.Maybe
import qualified MAlonzo.Code.Agda.Builtin.Unit
import qualified MAlonzo.Code.Agda.Primitive
import qualified MAlonzo.Code.Data.Bool.Base
import qualified MAlonzo.Code.Data.Sum.Base
import qualified MAlonzo.Code.Function.Base
import qualified MAlonzo.Code.IO.Primitive.Core
import qualified MAlonzo.Code.Level

-- IO.Base.IO
d_IO_20 a0 a1 = ()
data T_IO_20
  = C_lift_26 (MAlonzo.Code.Agda.Builtin.IO.T_IO_8 AgdaAny AgdaAny) |
    C_pure_30 AgdaAny |
    C_bind_40 (MAlonzo.RTE.Infinity AgdaAny T_IO_20)
              (AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20) |
    C_seq_48 (MAlonzo.RTE.Infinity AgdaAny T_IO_20)
             (MAlonzo.RTE.Infinity AgdaAny T_IO_20)
-- IO.Base.lift!
d_lift'33'_50 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> MAlonzo.Code.Agda.Primitive.T_Level_18 -> T_IO_20 -> T_IO_20
d_lift'33'_50 v0 ~v1 v2 v3 = du_lift'33'_50 v0 v2 v3
du_lift'33'_50 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 -> T_IO_20 -> T_IO_20
du_lift'33'_50 v0 v1 v2
  = case coe v2 of
      C_lift_26 v3
        -> coe
             C_lift_26
             (coe
                MAlonzo.Code.IO.Primitive.Core.d__'62''62''61'__14 v0 erased ()
                erased v3
                (\ v4 ->
                   coe
                     MAlonzo.Code.IO.Primitive.Core.d_pure_12 () erased
                     (coe MAlonzo.Code.Level.C_lift_20 (coe v4))))
      C_pure_30 v3
        -> coe C_pure_30 (coe MAlonzo.Code.Level.C_lift_20 (coe v3))
      C_bind_40 v4 v5
        -> coe
             C_bind_40 (coe du_'9839''45'0_1485 (coe v0) (coe v1) (coe v4))
             (coe du_'9839''45'1_1911 (coe v0) (coe v1) (coe v5))
      C_seq_48 v4 v5
        -> coe
             C_seq_48 (coe du_'9839''45'2_2737 (coe v0) (coe v1) (coe v4))
             (coe du_'9839''45'3_3153 (coe v0) (coe v1) (coe v5))
      _ -> MAlonzo.RTE.mazUnreachableError
-- IO.Base._._<*>_
d__'60''42''62'__82 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> () -> T_IO_20 -> T_IO_20 -> T_IO_20
d__'60''42''62'__82 ~v0 ~v1 ~v2 v3 v4 = du__'60''42''62'__82 v3 v4
du__'60''42''62'__82 :: T_IO_20 -> T_IO_20 -> T_IO_20
du__'60''42''62'__82 v0 v1
  = coe
      C_bind_40 (coe du_'9839''45'4_5787 (coe v0))
      (coe du_'9839''45'5_6021 (coe v1))
-- IO.Base._._<$>_
d__'60''36''62'__92 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> () -> (AgdaAny -> AgdaAny) -> T_IO_20 -> T_IO_20
d__'60''36''62'__92 ~v0 ~v1 ~v2 v3 v4 = du__'60''36''62'__92 v3 v4
du__'60''36''62'__92 :: (AgdaAny -> AgdaAny) -> T_IO_20 -> T_IO_20
du__'60''36''62'__92 v0 v1
  = coe du__'60''42''62'__82 (coe C_pure_30 (coe v0)) (coe v1)
-- IO.Base._._<$_
d__'60''36'__98 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> () -> AgdaAny -> T_IO_20 -> T_IO_20
d__'60''36'__98 ~v0 ~v1 ~v2 v3 v4 = du__'60''36'__98 v3 v4
du__'60''36'__98 :: AgdaAny -> T_IO_20 -> T_IO_20
du__'60''36'__98 v0 v1
  = coe du__'60''36''62'__92 (coe (\ v2 -> v0)) (coe v1)
-- IO.Base._._>>=_
d__'62''62''61'__104 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> () -> T_IO_20 -> (AgdaAny -> T_IO_20) -> T_IO_20
d__'62''62''61'__104 ~v0 ~v1 ~v2 v3 v4
  = du__'62''62''61'__104 v3 v4
du__'62''62''61'__104 :: T_IO_20 -> (AgdaAny -> T_IO_20) -> T_IO_20
du__'62''62''61'__104 v0 v1
  = coe
      C_bind_40 (coe du_'9839''45'8_7861 (coe v0))
      (coe du_'9839''45'9_8095 (coe v1))
-- IO.Base._._=<<_
d__'61''60''60'__112 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> () -> (AgdaAny -> T_IO_20) -> T_IO_20 -> T_IO_20
d__'61''60''60'__112 ~v0 ~v1 ~v2 v3 v4
  = du__'61''60''60'__112 v3 v4
du__'61''60''60'__112 :: (AgdaAny -> T_IO_20) -> T_IO_20 -> T_IO_20
du__'61''60''60'__112 v0 v1
  = coe du__'62''62''61'__104 (coe v1) (coe v0)
-- IO.Base._._>>_
d__'62''62'__114 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> () -> T_IO_20 -> T_IO_20 -> T_IO_20
d__'62''62'__114 ~v0 ~v1 ~v2 v3 v4 = du__'62''62'__114 v3 v4
du__'62''62'__114 :: T_IO_20 -> T_IO_20 -> T_IO_20
du__'62''62'__114 v0 v1
  = coe
      C_seq_48 (coe du_'9839''45'10_8907 (coe v0))
      (coe du_'9839''45'11_9131 (coe v1))
-- IO.Base._._<<_
d__'60''60'__120 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> () -> T_IO_20 -> T_IO_20 -> T_IO_20
d__'60''60'__120 ~v0 ~v1 ~v2 v3 v4 = du__'60''60'__120 v3 v4
du__'60''60'__120 :: T_IO_20 -> T_IO_20 -> T_IO_20
du__'60''60'__120 v0 v1 = coe du__'62''62'__114 (coe v1) (coe v0)
-- IO.Base.run
d_run_122 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  T_IO_20 -> MAlonzo.Code.Agda.Builtin.IO.T_IO_8 AgdaAny AgdaAny
d_run_122 v0 ~v1 v2 = du_run_122 v0 v2
du_run_122 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  T_IO_20 -> MAlonzo.Code.Agda.Builtin.IO.T_IO_8 AgdaAny AgdaAny
du_run_122 v0 v1
  = case coe v1 of
      C_lift_26 v2 -> coe v2
      C_pure_30 v2
        -> coe MAlonzo.Code.IO.Primitive.Core.d_pure_12 v0 erased v2
      C_bind_40 v3 v4
        -> coe
             MAlonzo.Code.IO.Primitive.Core.d__'62''62''61'__14 v0 erased v0
             erased
             (coe
                du_run_122 (coe v0)
                (coe MAlonzo.Code.Agda.Builtin.Coinduction.d_'9837'_22 (coe v3)))
             (\ v5 ->
                coe
                  du_run_122 (coe v0)
                  (coe
                     MAlonzo.Code.Agda.Builtin.Coinduction.d_'9837'_22 (coe v4 v5)))
      C_seq_48 v3 v4
        -> coe
             MAlonzo.Code.IO.Primitive.Core.d__'62''62''61'__14 v0 erased v0
             erased
             (coe
                du_run_122 (coe v0)
                (coe MAlonzo.Code.Agda.Builtin.Coinduction.d_'9837'_22 (coe v3)))
             (\ v5 ->
                coe
                  du_run_122 (coe v0)
                  (coe MAlonzo.Code.Agda.Builtin.Coinduction.d_'9837'_22 (coe v4)))
      _ -> MAlonzo.RTE.mazUnreachableError
-- IO.Base.Main
d_Main_140 :: ()
d_Main_140 = erased
-- IO.Base.lift′
d_lift'8242'_142 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Builtin.IO.T_IO_8
    AgdaAny MAlonzo.Code.Agda.Builtin.Unit.T_'8868'_6 ->
  T_IO_20
d_lift'8242'_142 v0 v1
  = coe
      C_lift_26
      (coe
         MAlonzo.Code.IO.Primitive.Core.d__'62''62''61'__14 () erased v0
         erased v1
         (\ v2 ->
            coe
              MAlonzo.Code.IO.Primitive.Core.d_pure_12 v0 erased
              (coe
                 MAlonzo.Code.Level.C_lift_20
                 (coe MAlonzo.Code.Agda.Builtin.Unit.C_tt_8))))
-- IO.Base.ignore
d_ignore_148 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 -> () -> T_IO_20 -> T_IO_20
d_ignore_148 ~v0 ~v1 v2 = du_ignore_148 v2
du_ignore_148 :: T_IO_20 -> T_IO_20
du_ignore_148 v0
  = coe
      du__'62''62'__114 (coe v0)
      (coe
         C_pure_30
         (coe
            MAlonzo.Code.Level.C_lift_20
            (coe MAlonzo.Code.Agda.Builtin.Unit.C_tt_8)))
-- IO.Base.when
d_when_152 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  Bool -> T_IO_20 -> T_IO_20
d_when_152 ~v0 v1 v2 = du_when_152 v1 v2
du_when_152 :: Bool -> T_IO_20 -> T_IO_20
du_when_152 v0 v1
  = if coe v0
      then coe v1
      else coe
             C_pure_30
             (coe
                MAlonzo.Code.Level.C_lift_20
                (coe MAlonzo.Code.Agda.Builtin.Unit.C_tt_8))
-- IO.Base.unless
d_unless_156 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  Bool -> T_IO_20 -> T_IO_20
d_unless_156 ~v0 = du_unless_156
du_unless_156 :: Bool -> T_IO_20 -> T_IO_20
du_unless_156
  = coe
      MAlonzo.Code.Function.Base.du__'8728''8242'__216 (coe du_when_152)
      (coe MAlonzo.Code.Data.Bool.Base.d_not_22)
-- IO.Base.whenJust
d_whenJust_158 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  Maybe AgdaAny -> (AgdaAny -> T_IO_20) -> T_IO_20
d_whenJust_158 ~v0 ~v1 ~v2 v3 v4 = du_whenJust_158 v3 v4
du_whenJust_158 :: Maybe AgdaAny -> (AgdaAny -> T_IO_20) -> T_IO_20
du_whenJust_158 v0 v1
  = case coe v0 of
      MAlonzo.Code.Agda.Builtin.Maybe.C_just_16 v2 -> coe v1 v2
      MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
        -> coe
             C_pure_30
             (coe
                MAlonzo.Code.Level.C_lift_20
                (coe MAlonzo.Code.Agda.Builtin.Unit.C_tt_8))
      _ -> MAlonzo.RTE.mazUnreachableError
-- IO.Base.whenInj₂
d_whenInj'8322'_164 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Data.Sum.Base.T__'8846'__30 ->
  (AgdaAny -> T_IO_20) -> T_IO_20
d_whenInj'8322'_164 ~v0 ~v1 ~v2 ~v3 ~v4 v5 v6
  = du_whenInj'8322'_164 v5 v6
du_whenInj'8322'_164 ::
  MAlonzo.Code.Data.Sum.Base.T__'8846'__30 ->
  (AgdaAny -> T_IO_20) -> T_IO_20
du_whenInj'8322'_164 v0 v1
  = case coe v0 of
      MAlonzo.Code.Data.Sum.Base.C_inj'8321'_38 v2
        -> coe
             C_pure_30
             (coe
                MAlonzo.Code.Level.C_lift_20
                (coe MAlonzo.Code.Agda.Builtin.Unit.C_tt_8))
      MAlonzo.Code.Data.Sum.Base.C_inj'8322'_42 v2 -> coe v1 v2
      _ -> MAlonzo.RTE.mazUnreachableError
-- IO.Base.forever
d_forever_170 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 -> T_IO_20 -> T_IO_20
d_forever_170 v0 v1
  = coe
      C_seq_48 (coe du_'9839''45'12_13737 (coe v1))
      (d_'9839''45'13_13829 (coe v0) (coe v1))
-- IO.Base.untilJust
d_untilJust_174 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 -> () -> T_IO_20 -> T_IO_20
d_untilJust_174 v0 ~v1 v2 = du_untilJust_174 v0 v2
du_untilJust_174 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 -> T_IO_20 -> T_IO_20
du_untilJust_174 v0 v1
  = coe
      C_bind_40 (coe du_'9839''45'14_14151 (coe v1))
      (coe du_'46'extendedlambda0_178 (coe v0) (coe v1))
-- IO.Base..extendedlambda0
d_'46'extendedlambda0_178 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  T_IO_20 -> Maybe AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'46'extendedlambda0_178 v0 ~v1 v2 v3
  = du_'46'extendedlambda0_178 v0 v2 v3
du_'46'extendedlambda0_178 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  T_IO_20 -> Maybe AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'46'extendedlambda0_178 v0 v1 v2
  = case coe v2 of
      MAlonzo.Code.Agda.Builtin.Maybe.C_just_16 v3
        -> coe du_'9839''45'16_14643 (coe v3)
      MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
        -> coe du_'9839''45'15_14373 (coe v0) (coe v1)
      _ -> MAlonzo.RTE.mazUnreachableError
-- IO.Base.untilInj₂
d_untilInj'8322'_186 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> () -> (AgdaAny -> T_IO_20) -> AgdaAny -> T_IO_20
d_untilInj'8322'_186 v0 ~v1 ~v2 v3 v4
  = du_untilInj'8322'_186 v0 v3 v4
du_untilInj'8322'_186 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (AgdaAny -> T_IO_20) -> AgdaAny -> T_IO_20
du_untilInj'8322'_186 v0 v1 v2
  = coe
      C_bind_40 (coe du_'9839''45'17_15611 (coe v1) (coe v2))
      (coe du_'46'extendedlambda1_192 (coe v0) (coe v1))
-- IO.Base..extendedlambda1
d_'46'extendedlambda1_192 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  (AgdaAny -> T_IO_20) ->
  AgdaAny ->
  MAlonzo.Code.Data.Sum.Base.T__'8846'__30 ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'46'extendedlambda1_192 v0 ~v1 ~v2 v3 ~v4 v5
  = du_'46'extendedlambda1_192 v0 v3 v5
du_'46'extendedlambda1_192 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (AgdaAny -> T_IO_20) ->
  MAlonzo.Code.Data.Sum.Base.T__'8846'__30 ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'46'extendedlambda1_192 v0 v1 v2
  = case coe v2 of
      MAlonzo.Code.Data.Sum.Base.C_inj'8321'_38 v3
        -> coe du_'9839''45'18_15999 (coe v0) (coe v1) (coe v3)
      MAlonzo.Code.Data.Sum.Base.C_inj'8322'_42 v3
        -> coe du_'9839''45'19_16509 (coe v3)
      _ -> MAlonzo.RTE.mazUnreachableError
-- IO.Base.untilRight
d_untilRight_198 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> () -> (AgdaAny -> T_IO_20) -> AgdaAny -> T_IO_20
d_untilRight_198 v0 v1 v2 v3 v4
  = coe du_untilInj'8322'_186 v0 v3 v4
-- IO.Base._.♯-0
d_'9839''45'0_1485 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20 ->
  (AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20) ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'0_1485 v0 ~v1 v2 ~v3 v4 ~v5
  = du_'9839''45'0_1485 v0 v2 v4
du_'9839''45'0_1485 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20 ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'0_1485 v0 v1 v2
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe
         du_lift'33'_50 (coe v0) (coe v1)
         (coe MAlonzo.Code.Agda.Builtin.Coinduction.d_'9837'_22 (coe v2)))
-- IO.Base._.♯-1
d_'9839''45'1_1911 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20 ->
  (AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20) ->
  MAlonzo.Code.Level.T_Lift_8 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'1_1911 v0 ~v1 v2 ~v3 ~v4 v5 v6
  = du_'9839''45'1_1911 v0 v2 v5 v6
du_'9839''45'1_1911 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20) ->
  MAlonzo.Code.Level.T_Lift_8 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'1_1911 v0 v1 v2 v3
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe
         du_lift'33'_50 (coe v0) (coe v1)
         (coe
            MAlonzo.Code.Agda.Builtin.Coinduction.d_'9837'_22
            (coe v2 (MAlonzo.Code.Level.d_lower_18 (coe v3)))))
-- IO.Base._.♯-2
d_'9839''45'2_2737 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20 ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20 ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'2_2737 v0 ~v1 v2 ~v3 v4 ~v5
  = du_'9839''45'2_2737 v0 v2 v4
du_'9839''45'2_2737 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20 ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'2_2737 v0 v1 v2
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe
         du_lift'33'_50 (coe v0) (coe v1)
         (coe MAlonzo.Code.Agda.Builtin.Coinduction.d_'9837'_22 (coe v2)))
-- IO.Base._.♯-3
d_'9839''45'3_3153 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20 ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20 ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'3_3153 v0 ~v1 v2 ~v3 ~v4 v5
  = du_'9839''45'3_3153 v0 v2 v5
du_'9839''45'3_3153 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20 ->
  MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'3_3153 v0 v1 v2
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe
         du_lift'33'_50 (coe v0) (coe v1)
         (coe MAlonzo.Code.Agda.Builtin.Coinduction.d_'9837'_22 (coe v2)))
-- IO.Base._._.♯-4
d_'9839''45'4_5787 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () -> T_IO_20 -> T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'4_5787 ~v0 ~v1 ~v2 v3 ~v4 = du_'9839''45'4_5787 v3
du_'9839''45'4_5787 ::
  T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'4_5787 v0
  = coe MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16 (coe v0)
-- IO.Base._._.♯-5
d_'9839''45'5_6021 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  T_IO_20 ->
  T_IO_20 ->
  (AgdaAny -> AgdaAny) -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'5_6021 ~v0 ~v1 ~v2 ~v3 v4 v5
  = du_'9839''45'5_6021 v4 v5
du_'9839''45'5_6021 ::
  T_IO_20 ->
  (AgdaAny -> AgdaAny) -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'5_6021 v0 v1
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe
         C_bind_40 (coe du_'9839''45'6_6281 (coe v0))
         (coe du_'9839''45'7_6595 (coe v1)))
-- IO.Base._._._.♯-6
d_'9839''45'6_6281 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  T_IO_20 ->
  T_IO_20 ->
  (AgdaAny -> AgdaAny) -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'6_6281 ~v0 ~v1 ~v2 ~v3 v4 ~v5 = du_'9839''45'6_6281 v4
du_'9839''45'6_6281 ::
  T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'6_6281 v0
  = coe MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16 (coe v0)
-- IO.Base._._._.♯-7
d_'9839''45'7_6595 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  T_IO_20 ->
  T_IO_20 ->
  (AgdaAny -> AgdaAny) ->
  AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'7_6595 ~v0 ~v1 ~v2 ~v3 ~v4 v5 v6
  = du_'9839''45'7_6595 v5 v6
du_'9839''45'7_6595 ::
  (AgdaAny -> AgdaAny) ->
  AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'7_6595 v0 v1
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe C_pure_30 (coe v0 v1))
-- IO.Base._._.♯-8
d_'9839''45'8_7861 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  T_IO_20 ->
  (AgdaAny -> T_IO_20) -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'8_7861 ~v0 ~v1 ~v2 v3 ~v4 = du_'9839''45'8_7861 v3
du_'9839''45'8_7861 ::
  T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'8_7861 v0
  = coe MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16 (coe v0)
-- IO.Base._._.♯-9
d_'9839''45'9_8095 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  T_IO_20 ->
  (AgdaAny -> T_IO_20) ->
  AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'9_8095 ~v0 ~v1 ~v2 ~v3 v4 v5
  = du_'9839''45'9_8095 v4 v5
du_'9839''45'9_8095 ::
  (AgdaAny -> T_IO_20) ->
  AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'9_8095 v0 v1
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16 (coe v0 v1)
-- IO.Base._._.♯-10
d_'9839''45'10_8907 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () -> T_IO_20 -> T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'10_8907 ~v0 ~v1 ~v2 v3 ~v4 = du_'9839''45'10_8907 v3
du_'9839''45'10_8907 ::
  T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'10_8907 v0
  = coe MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16 (coe v0)
-- IO.Base._._.♯-11
d_'9839''45'11_9131 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () -> T_IO_20 -> T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'11_9131 ~v0 ~v1 ~v2 ~v3 v4 = du_'9839''45'11_9131 v4
du_'9839''45'11_9131 ::
  T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'11_9131 v0
  = coe MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16 (coe v0)
-- IO.Base._.♯-12
d_'9839''45'12_13737 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'12_13737 ~v0 v1 = du_'9839''45'12_13737 v1
du_'9839''45'12_13737 ::
  T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'12_13737 v0
  = coe MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16 (coe v0)
-- IO.Base._.♯-13
d_'9839''45'13_13829 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'13_13829 v0 v1
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe d_forever_170 (coe v0) (coe v1))
-- IO.Base._.♯-14
d_'9839''45'14_14151 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'14_14151 ~v0 ~v1 v2 = du_'9839''45'14_14151 v2
du_'9839''45'14_14151 ::
  T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'14_14151 v0
  = coe MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16 (coe v0)
-- IO.Base._.♯-15
d_'9839''45'15_14373 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'15_14373 v0 ~v1 v2 = du_'9839''45'15_14373 v0 v2
du_'9839''45'15_14373 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  T_IO_20 -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'15_14373 v0 v1
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe du_untilJust_174 (coe v0) (coe v1))
-- IO.Base._.♯-16
d_'9839''45'16_14643 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () -> T_IO_20 -> AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'16_14643 ~v0 ~v1 ~v2 v3 = du_'9839''45'16_14643 v3
du_'9839''45'16_14643 ::
  AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'16_14643 v0
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe C_pure_30 (coe v0))
-- IO.Base._.♯-17
d_'9839''45'17_15611 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  (AgdaAny -> T_IO_20) ->
  AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'17_15611 ~v0 ~v1 ~v2 v3 v4
  = du_'9839''45'17_15611 v3 v4
du_'9839''45'17_15611 ::
  (AgdaAny -> T_IO_20) ->
  AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'17_15611 v0 v1
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16 (coe v0 v1)
-- IO.Base._.♯-18
d_'9839''45'18_15999 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  (AgdaAny -> T_IO_20) ->
  AgdaAny -> AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'18_15999 v0 ~v1 ~v2 v3 ~v4 v5
  = du_'9839''45'18_15999 v0 v3 v5
du_'9839''45'18_15999 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (AgdaAny -> T_IO_20) ->
  AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'18_15999 v0 v1 v2
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe du_untilInj'8322'_186 (coe v0) (coe v1) (coe v2))
-- IO.Base._.♯-19
d_'9839''45'19_16509 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  () ->
  (AgdaAny -> T_IO_20) ->
  AgdaAny -> AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
d_'9839''45'19_16509 ~v0 ~v1 ~v2 ~v3 ~v4 v5
  = du_'9839''45'19_16509 v5
du_'9839''45'19_16509 ::
  AgdaAny -> MAlonzo.RTE.Infinity AgdaAny T_IO_20
du_'9839''45'19_16509 v0
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe C_pure_30 (coe v0))

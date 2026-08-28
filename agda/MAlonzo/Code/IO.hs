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

module MAlonzo.Code.IO where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.Coinduction
import qualified MAlonzo.Code.Agda.Builtin.List
import qualified MAlonzo.Code.Agda.Builtin.Unit
import qualified MAlonzo.Code.Agda.Primitive
import qualified MAlonzo.Code.Codata.Musical.Colist.Base
import qualified MAlonzo.Code.Data.List.Base
import qualified MAlonzo.Code.IO.Base
import qualified MAlonzo.Code.Level

-- IO.Colist.sequence
d_sequence_12 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.Code.IO.Base.T_IO_20
d_sequence_12 v0 ~v1 v2 = du_sequence_12 v0 v2
du_sequence_12 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.Code.IO.Base.T_IO_20
du_sequence_12 v0 v1
  = case coe v1 of
      MAlonzo.Code.Codata.Musical.Colist.Base.C_'91''93'_24
        -> coe MAlonzo.Code.IO.Base.C_pure_30 (coe v1)
      MAlonzo.Code.Codata.Musical.Colist.Base.C__'8759'__30 v2 v3
        -> coe
             MAlonzo.Code.IO.Base.C_bind_40 (coe du_'9839''45'0_327 (coe v2))
             (coe du_'9839''45'1_517 (coe v0) (coe v3))
      _ -> MAlonzo.RTE.mazUnreachableError
-- IO.Colist.sequence′
d_sequence'8242'_22 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.Code.IO.Base.T_IO_20
d_sequence'8242'_22 v0 ~v1 v2 = du_sequence'8242'_22 v0 v2
du_sequence'8242'_22 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.Code.IO.Base.T_IO_20
du_sequence'8242'_22 v0 v1
  = case coe v1 of
      MAlonzo.Code.Codata.Musical.Colist.Base.C_'91''93'_24
        -> coe
             MAlonzo.Code.IO.Base.C_pure_30
             (coe
                MAlonzo.Code.Level.C_lift_20
                (coe MAlonzo.Code.Agda.Builtin.Unit.C_tt_8))
      MAlonzo.Code.Codata.Musical.Colist.Base.C__'8759'__30 v2 v3
        -> coe
             MAlonzo.Code.IO.Base.C_seq_48 (coe du_'9839''45'5_2749 (coe v2))
             (coe du_'9839''45'6_2937 (coe v0) (coe v3))
      _ -> MAlonzo.RTE.mazUnreachableError
-- IO.Colist.mapM
d_mapM_28 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny AgdaAny ->
  MAlonzo.Code.IO.Base.T_IO_20
d_mapM_28 v0 ~v1 v2 ~v3 v4 v5 = du_mapM_28 v0 v2 v4 v5
du_mapM_28 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny AgdaAny ->
  MAlonzo.Code.IO.Base.T_IO_20
du_mapM_28 v0 v1 v2 v3
  = coe
      du_sequence_12 (coe v1)
      (coe
         MAlonzo.Code.Codata.Musical.Colist.Base.du_map_40 (coe v0) (coe ())
         (coe v2) (coe v3))
-- IO.Colist.mapM′
d_mapM'8242'_32 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny AgdaAny ->
  MAlonzo.Code.IO.Base.T_IO_20
d_mapM'8242'_32 v0 ~v1 v2 ~v3 v4 v5 = du_mapM'8242'_32 v0 v2 v4 v5
du_mapM'8242'_32 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny AgdaAny ->
  MAlonzo.Code.IO.Base.T_IO_20
du_mapM'8242'_32 v0 v1 v2 v3
  = coe
      du_sequence'8242'_22 (coe v1)
      (coe
         MAlonzo.Code.Codata.Musical.Colist.Base.du_map_40 (coe v0) (coe ())
         (coe v2) (coe v3))
-- IO.Colist.forM
d_forM_36 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny AgdaAny ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.Code.IO.Base.T_IO_20
d_forM_36 v0 ~v1 v2 ~v3 v4 v5 = du_forM_36 v0 v2 v4 v5
du_forM_36 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny AgdaAny ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.Code.IO.Base.T_IO_20
du_forM_36 v0 v1 v2 v3
  = coe du_mapM_28 (coe v0) (coe v1) (coe v3) (coe v2)
-- IO.Colist.forM′
d_forM'8242'_38 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny AgdaAny ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.Code.IO.Base.T_IO_20
d_forM'8242'_38 v0 ~v1 v2 ~v3 v4 v5 = du_forM'8242'_38 v0 v2 v4 v5
du_forM'8242'_38 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny AgdaAny ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.Code.IO.Base.T_IO_20
du_forM'8242'_38 v0 v1 v2 v3
  = coe du_mapM'8242'_32 (coe v0) (coe v1) (coe v3) (coe v2)
-- IO.List.sequence
d_sequence_42 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  [MAlonzo.Code.IO.Base.T_IO_20] -> MAlonzo.Code.IO.Base.T_IO_20
d_sequence_42 ~v0 ~v1 v2 = du_sequence_42 v2
du_sequence_42 ::
  [MAlonzo.Code.IO.Base.T_IO_20] -> MAlonzo.Code.IO.Base.T_IO_20
du_sequence_42 v0
  = case coe v0 of
      [] -> coe MAlonzo.Code.IO.Base.C_pure_30 (coe v0)
      (:) v1 v2
        -> coe
             MAlonzo.Code.IO.Base.du__'60''42''62'__82
             (coe
                MAlonzo.Code.IO.Base.du__'60''42''62'__82
                (coe
                   MAlonzo.Code.IO.Base.C_pure_30
                   (coe MAlonzo.Code.Agda.Builtin.List.C__'8759'__22))
                (coe v1))
             (coe du_sequence_42 (coe v2))
      _ -> MAlonzo.RTE.mazUnreachableError
-- IO.List.sequence′
d_sequence'8242'_48 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  [MAlonzo.Code.IO.Base.T_IO_20] -> MAlonzo.Code.IO.Base.T_IO_20
d_sequence'8242'_48 ~v0 ~v1 v2 = du_sequence'8242'_48 v2
du_sequence'8242'_48 ::
  [MAlonzo.Code.IO.Base.T_IO_20] -> MAlonzo.Code.IO.Base.T_IO_20
du_sequence'8242'_48 v0
  = case coe v0 of
      []
        -> coe
             MAlonzo.Code.IO.Base.C_pure_30
             (coe
                MAlonzo.Code.Level.C_lift_20
                (coe MAlonzo.Code.Agda.Builtin.Unit.C_tt_8))
      (:) v1 v2
        -> coe
             MAlonzo.Code.IO.Base.du__'62''62'__114 (coe v1)
             (coe du_sequence'8242'_48 (coe v2))
      _ -> MAlonzo.RTE.mazUnreachableError
-- IO.List.mapM
d_mapM_54 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  [AgdaAny] -> MAlonzo.Code.IO.Base.T_IO_20
d_mapM_54 ~v0 ~v1 ~v2 ~v3 v4 v5 = du_mapM_54 v4 v5
du_mapM_54 ::
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  [AgdaAny] -> MAlonzo.Code.IO.Base.T_IO_20
du_mapM_54 v0 v1
  = coe
      du_sequence_42
      (coe MAlonzo.Code.Data.List.Base.du_map_22 (coe v0) (coe v1))
-- IO.List.mapM′
d_mapM'8242'_58 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  [AgdaAny] -> MAlonzo.Code.IO.Base.T_IO_20
d_mapM'8242'_58 ~v0 ~v1 ~v2 ~v3 v4 v5 = du_mapM'8242'_58 v4 v5
du_mapM'8242'_58 ::
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  [AgdaAny] -> MAlonzo.Code.IO.Base.T_IO_20
du_mapM'8242'_58 v0 v1
  = coe
      du_sequence'8242'_48
      (coe MAlonzo.Code.Data.List.Base.du_map_22 (coe v0) (coe v1))
-- IO.List.forM
d_forM_62 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  [AgdaAny] ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.Code.IO.Base.T_IO_20
d_forM_62 ~v0 ~v1 ~v2 ~v3 v4 v5 = du_forM_62 v4 v5
du_forM_62 ::
  [AgdaAny] ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.Code.IO.Base.T_IO_20
du_forM_62 v0 v1 = coe du_mapM_54 (coe v1) (coe v0)
-- IO.List.forM′
d_forM'8242'_64 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  [AgdaAny] ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.Code.IO.Base.T_IO_20
d_forM'8242'_64 ~v0 ~v1 ~v2 ~v3 v4 v5 = du_forM'8242'_64 v4 v5
du_forM'8242'_64 ::
  [AgdaAny] ->
  (AgdaAny -> MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.Code.IO.Base.T_IO_20
du_forM'8242'_64 v0 v1 = coe du_mapM'8242'_58 (coe v1) (coe v0)
-- IO.Colist._.♯-0
d_'9839''45'0_327 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.RTE.Infinity
    AgdaAny
    (MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
       AgdaAny MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.RTE.Infinity AgdaAny MAlonzo.Code.IO.Base.T_IO_20
d_'9839''45'0_327 ~v0 ~v1 v2 ~v3 = du_'9839''45'0_327 v2
du_'9839''45'0_327 ::
  MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.RTE.Infinity AgdaAny MAlonzo.Code.IO.Base.T_IO_20
du_'9839''45'0_327 v0
  = coe MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16 (coe v0)
-- IO.Colist._.♯-1
d_'9839''45'1_517 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.RTE.Infinity
    AgdaAny
    (MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
       AgdaAny MAlonzo.Code.IO.Base.T_IO_20) ->
  AgdaAny ->
  MAlonzo.RTE.Infinity AgdaAny MAlonzo.Code.IO.Base.T_IO_20
d_'9839''45'1_517 v0 ~v1 ~v2 v3 v4 = du_'9839''45'1_517 v0 v3 v4
du_'9839''45'1_517 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.RTE.Infinity
    AgdaAny
    (MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
       AgdaAny MAlonzo.Code.IO.Base.T_IO_20) ->
  AgdaAny ->
  MAlonzo.RTE.Infinity AgdaAny MAlonzo.Code.IO.Base.T_IO_20
du_'9839''45'1_517 v0 v1 v2
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe
         MAlonzo.Code.IO.Base.C_bind_40
         (coe du_'9839''45'2_735 (coe v0) (coe v1))
         (coe du_'9839''45'3_1123 (coe v2)))
-- IO.Colist._._.♯-2
d_'9839''45'2_735 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.RTE.Infinity
    AgdaAny
    (MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
       AgdaAny MAlonzo.Code.IO.Base.T_IO_20) ->
  AgdaAny ->
  MAlonzo.RTE.Infinity AgdaAny MAlonzo.Code.IO.Base.T_IO_20
d_'9839''45'2_735 v0 ~v1 ~v2 v3 ~v4 = du_'9839''45'2_735 v0 v3
du_'9839''45'2_735 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.RTE.Infinity
    AgdaAny
    (MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
       AgdaAny MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.RTE.Infinity AgdaAny MAlonzo.Code.IO.Base.T_IO_20
du_'9839''45'2_735 v0 v1
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe
         du_sequence_12 (coe v0)
         (coe MAlonzo.Code.Agda.Builtin.Coinduction.d_'9837'_22 (coe v1)))
-- IO.Colist._._.♯-3
d_'9839''45'3_1123 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.RTE.Infinity
    AgdaAny
    (MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
       AgdaAny MAlonzo.Code.IO.Base.T_IO_20) ->
  AgdaAny ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny AgdaAny ->
  MAlonzo.RTE.Infinity AgdaAny MAlonzo.Code.IO.Base.T_IO_20
d_'9839''45'3_1123 ~v0 ~v1 ~v2 ~v3 v4 v5
  = du_'9839''45'3_1123 v4 v5
du_'9839''45'3_1123 ::
  AgdaAny ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny AgdaAny ->
  MAlonzo.RTE.Infinity AgdaAny MAlonzo.Code.IO.Base.T_IO_20
du_'9839''45'3_1123 v0 v1
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe
         MAlonzo.Code.IO.Base.C_pure_30
         (coe
            MAlonzo.Code.Codata.Musical.Colist.Base.C__'8759'__30 (coe v0)
            (coe du_'9839''45'4_1457 (coe v1))))
-- IO.Colist._._._.♯-4
d_'9839''45'4_1457 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.RTE.Infinity
    AgdaAny
    (MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
       AgdaAny MAlonzo.Code.IO.Base.T_IO_20) ->
  AgdaAny ->
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny AgdaAny ->
  MAlonzo.RTE.Infinity
    AgdaAny
    (MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
       AgdaAny AgdaAny)
d_'9839''45'4_1457 ~v0 ~v1 ~v2 ~v3 ~v4 v5 = du_'9839''45'4_1457 v5
du_'9839''45'4_1457 ::
  MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
    AgdaAny AgdaAny ->
  MAlonzo.RTE.Infinity
    AgdaAny
    (MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
       AgdaAny AgdaAny)
du_'9839''45'4_1457 v0
  = coe MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16 (coe v0)
-- IO.Colist._.♯-5
d_'9839''45'5_2749 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.RTE.Infinity
    AgdaAny
    (MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
       AgdaAny MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.RTE.Infinity AgdaAny MAlonzo.Code.IO.Base.T_IO_20
d_'9839''45'5_2749 ~v0 ~v1 v2 ~v3 = du_'9839''45'5_2749 v2
du_'9839''45'5_2749 ::
  MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.RTE.Infinity AgdaAny MAlonzo.Code.IO.Base.T_IO_20
du_'9839''45'5_2749 v0
  = coe MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16 (coe v0)
-- IO.Colist._.♯-6
d_'9839''45'6_2937 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  () ->
  MAlonzo.Code.IO.Base.T_IO_20 ->
  MAlonzo.RTE.Infinity
    AgdaAny
    (MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
       AgdaAny MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.RTE.Infinity AgdaAny MAlonzo.Code.IO.Base.T_IO_20
d_'9839''45'6_2937 v0 ~v1 ~v2 v3 = du_'9839''45'6_2937 v0 v3
du_'9839''45'6_2937 ::
  MAlonzo.Code.Agda.Primitive.T_Level_18 ->
  MAlonzo.RTE.Infinity
    AgdaAny
    (MAlonzo.Code.Codata.Musical.Colist.Base.T_Colist_20
       AgdaAny MAlonzo.Code.IO.Base.T_IO_20) ->
  MAlonzo.RTE.Infinity AgdaAny MAlonzo.Code.IO.Base.T_IO_20
du_'9839''45'6_2937 v0 v1
  = coe
      MAlonzo.Code.Agda.Builtin.Coinduction.C_'9839'__16
      (coe
         du_sequence'8242'_22 (coe v0)
         (coe MAlonzo.Code.Agda.Builtin.Coinduction.d_'9837'_22 (coe v1)))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Из GeoJSON (EPSG:4326) собираем raw_geo для GlobalMapper-пайплайна.
По одному .pkl на квартал: {
  'block': shapely.Polygon (в UTM),
  'buildings': [Polygon, ...] (в UTM),
  'zone': <лейбл функциональной зоны>
}

Особенности:
- Быстрое предварительное отсечение зданий по bbox (sjoin)
- Гибкий способ выбора зданий: по центроиду или по пересечению
- Валидация геометрии (buffer(0)), распад MultiPolygon → Polygons
- Пропуск кварталов без зданий
- Автоматический выбор UTM по центроиду квартала
"""

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import shapely
import pickle
import pathlib
import math
import warnings
warnings.filterwarnings("ignore")

# ===================== НАСТРОЙКИ =====================
BLOCKS_PATH = "data/zones.geojson"        # вход: кварталы (EPSG:4326), должны содержать поле 'zone'
BUILDINGS_PATH = "data/buildings.geojson"  # вход: здания (EPSG:4326)
OUT_DIR = pathlib.Path("my_dataset/raw_geo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Как отбирать здания для квартала: 'centroid' (быстрее, стабильно) или 'intersection' (строже)
BUILDING_SELECT_MODE = "centroid"  # 'centroid' | 'intersection'

# Минимальная площадь здания в м² после перехода в UTM
MIN_BUILDING_AREA = 5.0

# Поле с лейблом зоны в GeoJSON кварталов
ZONE_FIELD = "functional_zone_type_name"

# =====================================================

def utm_epsg_for(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def _make_valid(g):
    try:
        if g is None or g.is_empty:
            return None
        # shapely >=2.0: make_valid доступен, но buffer(0) остаётся универсальным подходом
        g2 = shapely.make_valid(g) if hasattr(shapely, "make_valid") else g.buffer(0)
        if g2 is None or g2.is_empty:
            return None
        return g2
    except Exception:
        return None


def _largest_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        # взять крупнейший по площади
        parts = [p for p in geom.geoms if isinstance(p, Polygon)]
        return max(parts, key=lambda p: p.area) if parts else None
    # прочие типы — проигнорируем
    return None


def load_inputs():
    blks = gpd.read_file(BLOCKS_PATH).to_crs(4326)
    bldg = gpd.read_file(BUILDINGS_PATH).to_crs(4326)

    # Валидация кварталов: MultiPolygon → крупнейший Polygon
    blks["geometry"] = blks["geometry"].apply(_make_valid).apply(_largest_polygon)
    blks = blks[blks["geometry"].notna()].copy()

    # Здания: чистим и оставляем только Polygon/MultiPolygon
    bldg["geometry"] = bldg["geometry"].apply(_make_valid)
    bldg = bldg[bldg["geometry"].notna()].copy()

    return blks, bldg


def buildings_in_block(block_row, buildings_gdf):
    blk_geom_wgs = block_row.geometry

    if BUILDING_SELECT_MODE == "centroid":
        # centroid-within (устойчиво к микропересечениям)
        centroids = buildings_gdf.copy()
        centroids["geometry"] = centroids["geometry"].centroid
        try:
            join = gpd.sjoin(centroids, gpd.GeoDataFrame(geometry=[blk_geom_wgs], crs=4326), predicate="within", how="inner")
            subset = buildings_gdf.loc[join.index].copy()
        except Exception:
            # fallback: bbox фильтр
            subset = buildings_gdf[buildings_gdf.intersects(blk_geom_wgs.buffer(0))]
    else:
        # строгая пересечённость
        try:
            subset = gpd.overlay(buildings_gdf, gpd.GeoDataFrame(geometry=[blk_geom_wgs], crs=4326), how="intersection", keep_geom_type=True)
        except Exception:
            subset = buildings_gdf[buildings_gdf.intersects(blk_geom_wgs.buffer(0))]

    return subset


def to_utm_series(geom_series, epsg):
    return geom_series.to_crs(epsg)


def main():
    blks, bldg = load_inputs()
    total, written, skipped_no_bldg, skipped_no_zone = 0, 0, 0, 0

    for i, blk in blks.iterrows():
        total += 1
        zone_lbl = blk.get(ZONE_FIELD, None)
        if zone_lbl is None or (isinstance(zone_lbl, float) and math.isnan(zone_lbl)):
            skipped_no_zone += 1
            continue

        # Собираем здания, относящиеся к кварталу (в 4326)
        b_in_wgs = buildings_in_block(blk, bldg)
        if b_in_wgs.empty:
            skipped_no_bldg += 1
            continue

        # На случай intersection-режима: гарантируем клип к границе квартала
        if BUILDING_SELECT_MODE == "centroid":
            try:
                b_in_wgs = gpd.overlay(b_in_wgs, gpd.GeoDataFrame(geometry=[blk.geometry], crs=4326), how="intersection", keep_geom_type=True)
            except Exception:
                pass
        if b_in_wgs.empty:
            skipped_no_bldg += 1
            continue

        # Выбираем UTM по центроиду квартала
        ctr = blk.geometry.centroid
        epsg = utm_epsg_for(ctr.x, ctr.y)

        # Проецируем квартал и здания в UTM
        blk_utm = to_utm_series(gpd.GeoSeries([blk.geometry], crs=4326), epsg).iloc[0]
        b_in_utm = to_utm_series(gpd.GeoSeries(b_in_wgs.geometry.values, crs=4326), epsg)

        # Раскладываем MultiPolygon → Polygon, фильтруем по площади и валидности
        poly_list = []
        for geom in b_in_utm:
            g_valid = _make_valid(geom)
            if g_valid is None:
                continue
            if isinstance(g_valid, Polygon):
                parts = [g_valid]
            elif isinstance(g_valid, MultiPolygon):
                parts = [p for p in g_valid.geoms if isinstance(p, Polygon)]
            else:
                parts = []
            for p in parts:
                if p.is_valid and p.area >= MIN_BUILDING_AREA:
                    poly_list.append(p)

        if not poly_list:
            skipped_no_bldg += 1
            continue

        payload = {
            "block": blk_utm,
            "buildings": poly_list,
            "zone": zone_lbl,
        }
        out_path = OUT_DIR / f"block_{i:06d}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        written += 1

    print(f"Всего кварталов: {total}")
    print(f"Записано pkl:    {written}")
    print(f"Скип без зданий: {skipped_no_bldg}")
    print(f"Скип без зоны:   {skipped_no_zone}")


if __name__ == "__main__":
    main()

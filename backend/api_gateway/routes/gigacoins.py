"""GigaCoin shop API: server-validated ownership, purchases and equipment."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway.dependecies import get_current_user
from backend.db.session import get_db
from backend.models.gigacoin_event import GigacoinEvent
from backend.models.user import User
from backend.models.user_cosmetic import UserCosmetic
from backend.services.cosmetics import COSMETICS

router = APIRouter(prefix="/api/gigacoins", tags=["gigacoins"])


class PurchaseRequest(BaseModel):
    cosmetic_id: str = Field(min_length=1, max_length=64)


class EquipRequest(BaseModel):
    cosmetic_id: str = Field(min_length=1, max_length=64)


def _item_payload(item_id: str, user: User, owned: set[str]) -> dict:
    item = COSMETICS[item_id]
    equipped_id = (
        user.equipped_avatar_frame if item["slot"] == "avatar_frame"
        else user.equipped_profile_accent
    )
    return {
        "id": item_id,
        **item,
        "owned": item_id in owned,
        "equipped": equipped_id == item_id,
    }


@router.get("/shop")
async def shop_catalog(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    rows = (await db.scalars(select(UserCosmetic).where(UserCosmetic.user_id == user.id))).all()
    owned = {row.cosmetic_id for row in rows}
    return {
        "balance": int(user.gigacoins or 0),
        "items": [_item_payload(item_id, user, owned) for item_id in COSMETICS],
    }


@router.post("/shop/purchase")
async def purchase_cosmetic(
    payload: PurchaseRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    item = COSMETICS.get(payload.cosmetic_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Предмет не найден")

    locked_user = await db.scalar(select(User).where(User.id == user.id).with_for_update())
    if locked_user is None:
        raise HTTPException(status_code=401, detail="Пользователь не найден")
    already_owned = await db.scalar(select(UserCosmetic.id).where(
        UserCosmetic.user_id == user.id,
        UserCosmetic.cosmetic_id == payload.cosmetic_id,
    ).limit(1))
    if already_owned is not None:
        raise HTTPException(status_code=409, detail="Предмет уже куплен")
    if int(locked_user.gigacoins or 0) < item["price"]:
        raise HTTPException(status_code=409, detail="Недостаточно GigaCoin")

    locked_user.gigacoins -= item["price"]
    db.add(UserCosmetic(user_id=user.id, cosmetic_id=payload.cosmetic_id))
    db.add(GigacoinEvent(
        user_id=user.id,
        reason="shop_purchase",
        ref=payload.cosmetic_id,
        amount=-item["price"],
    ))
    await db.commit()
    return {
        "ok": True,
        "balance": int(locked_user.gigacoins or 0),
        "user": locked_user.public(),
        "item": _item_payload(payload.cosmetic_id, locked_user, {payload.cosmetic_id}),
    }


@router.post("/shop/equip")
async def equip_cosmetic(
    payload: EquipRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    item = COSMETICS.get(payload.cosmetic_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Предмет не найден")
    user = await db.scalar(select(User).where(User.id == user.id).with_for_update())
    if user is None:
        raise HTTPException(status_code=401, detail="Пользователь не найден")
    owned = await db.scalar(select(UserCosmetic.id).where(
        UserCosmetic.user_id == user.id,
        UserCosmetic.cosmetic_id == payload.cosmetic_id,
    ).limit(1))
    if owned is None:
        raise HTTPException(status_code=403, detail="Сначала купите предмет")

    slot = item["slot"]
    await db.execute(
        update(UserCosmetic)
        .where(UserCosmetic.user_id == user.id, UserCosmetic.cosmetic_id.in_(
            [key for key, value in COSMETICS.items() if value["slot"] == slot]
        ))
        .values(is_equipped=False)
    )
    row = await db.scalar(select(UserCosmetic).where(
        UserCosmetic.user_id == user.id,
        UserCosmetic.cosmetic_id == payload.cosmetic_id,
    ))
    row.is_equipped = True
    if slot == "avatar_frame":
        user.equipped_avatar_frame = payload.cosmetic_id
    else:
        user.equipped_profile_accent = payload.cosmetic_id
    await db.commit()
    return {"ok": True, "user": user.public(), "cosmetic_id": payload.cosmetic_id}

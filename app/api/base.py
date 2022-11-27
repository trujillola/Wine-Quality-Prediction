from typing import List, Optional
from fastapi import APIRouter, Header


router = APIRouter(
    prefix='/api',
    tags = ['api']
)
async def read_items(x_token : Optional[List[str]] = Header(None)):
    return {"X-Token values" : x_token}
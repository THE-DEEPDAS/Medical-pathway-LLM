from pydantic import BaseModel
from typing import Optional

class OptimizerConfig(BaseModel):
    max_optimization_threads: Optional[int] = 4
    
    class Config:
        extra = "allow"

class StrictModeConfig(BaseModel):
    enabled: bool = False
    
    class Config:
        extra = "allow"

class QdrantConfig(BaseModel):
    optimizer_config: OptimizerConfig = OptimizerConfig()
    strict_mode_config: StrictModeConfig = StrictModeConfig()
    
    class Config:
        extra = "allow"

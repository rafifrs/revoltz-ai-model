from pydantic import BaseModel, ConfigDict, Field, computed_field


class CellInput(BaseModel):
    cell_id: str = Field(..., min_length=1, description="Unique cell identifier from workshop")
    IR_ohm: float = Field(..., gt=0, description="Measured internal resistance in ohm")
    OCV_V: float = Field(..., gt=0, description="Measured open circuit voltage in volt")
    chemistry: str | None = Field(default=None, description="Optional chemistry label, if known")
    source_pack_id: str | None = Field(default=None, description="Optional source battery pack identifier")
    source_battery_id: str | None = Field(default=None, description="Optional source battery identifier")
    position_in_pack: str | None = Field(default=None, description="Optional source cell position inside the original pack")


class CellResult(BaseModel):
    cell_id: str
    IR_ohm: float
    OCV_V: float
    chemistry: str | None = None
    source_pack_id: str | None = None
    source_battery_id: str | None = None
    position_in_pack: str | None = None
    is_outlier: bool
    cluster: int | None
    compatible_with: list[str]
    recommended_pack_id: str | None = None
    status: str
    notes: list[str]


class AssemblyRequest(BaseModel):
    target_pack_size: int = Field(..., ge=2, le=512, description="Desired number of cells in each recommended repack group")
    max_packs: int | None = Field(default=None, ge=1, le=100, description="Optional cap on how many pack recommendations to return")
    allow_partial_packs: bool = Field(default=True, description="Whether to return partial candidate groups when a full pack cannot be formed")


class PackRecommendation(BaseModel):
    pack_id: str
    status: str
    cluster: int | None
    cell_ids: list[str]
    source_pack_ids: list[str]
    target_pack_size: int
    cell_count: int
    missing_cells: int
    average_ir_ohm: float
    average_ocv_v: float
    average_pairwise_delta_ir: float
    average_pairwise_delta_ocv: float
    homogeneity_score: float
    notes: list[str]


class PredictRequest(BaseModel):
    cells: list[CellInput] = Field(..., min_length=1, max_length=500)
    assembly_request: AssemblyRequest | None = None


class PredictSummary(BaseModel):
    total_cells: int
    outlier_count: int
    valid_for_repack: int
    clusters_found: int
    compatible_pairs: int
    recommended_pack_count: int
    status_breakdown: dict[str, int]

    @computed_field
    @property
    def repack_rate(self) -> float:
        if self.total_cells == 0:
            return 0.0
        return self.valid_for_repack / self.total_cells


class PredictResponse(BaseModel):
    results: list[CellResult]
    summary: PredictSummary
    recommended_packs: list[PackRecommendation] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    selected_clusters: int
    snapshot_cycle: int | None
    training_samples: int
    non_outlier_training_samples: int
    compatibility_thresholds: dict[str, float]
    iqr_bounds: dict[str, tuple[float, float]]
    silhouette_score: float | None


class PackInput(BaseModel):
    pack_id: str = Field(..., min_length=1)
    ocv_v: float = Field(..., gt=0)
    capacity_ah: float = Field(..., gt=0)
    cycle_count: int = Field(..., ge=0)
    temperature_c: float = Field(..., gt=-100, lt=200)
    age_days: float | None = None
    chemistry: str | None = None


class PackPredictionResult(BaseModel):
    pack_id: str
    predicted_soh: float
    confidence_score: float
    recommended_action: str
    notes: list[str]


class PredictPackRequest(BaseModel):
    packs: list[PackInput] = Field(..., min_length=1, max_length=500)


class PredictPackResponse(BaseModel):
    results: list[PackPredictionResult]
    summary: dict[str, float | int]


class Model1InfoResponse(BaseModel):
    model_version: str
    training_rows: int
    eval_rows: int
    mae: float
    rmse: float
    r2: float

"""L3 gating accepts the released frame paths, IDs and answer fields."""

from drivestreambench.metrics.l3_gated import (
    build_video_canonical_correct,
    video_clip_id_from_qid,
    video_id_from_record,
)

VIDEO = "2021.01.01.10.00.00_veh-01_0001_0100"


def test_relative_release_frame_path_resolves_video():
    record = {
        "question_id": "L3_other",
        "image_samples": [f"nuplan-v1.1_train_camera_0/{VIDEO}/CAM_F0/frame.jpg"],
    }
    assert video_id_from_record(record) == VIDEO


def test_release_qid_level_prefix_is_supported():
    assert (
        video_clip_id_from_qid(f"L4_{VIDEO}_lead_vehicle_start_test_positive_0001")
        == VIDEO
    )


def test_gold_answer_format_controls_mcq_detection():
    common = {"image_samples": [f"nuplan-v1.1_train_camera_0/{VIDEO}/CAM_F0/frame.jpg"]}
    records = [
        {
            **common,
            "question_id": "L1_0001",
            "question_level": "L1",
            "answer": "A",
            "question_type": "vehicle",
        },
        {
            **common,
            "question_id": "L2_0001",
            "question_level": "L2",
            "answer": "YES",
            "question_type": "ego_lane",
        },
    ]
    result = build_video_canonical_correct(
        records, [{"prediction": "A"}, {"prediction": "YES"}]
    )
    assert result[VIDEO] == {"L1": True, "L2": True}

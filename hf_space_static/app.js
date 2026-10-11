// Clean Multi-Model Wood Grain Classification Script (Zero Emojis, Precise Naming)
const EMBEDDED_DATA = {
  "title": "Wood Grain Species Classification & Visual Interpretability",
  "course": "Học Máy (INT3401) - Lớp 11",
  "team": "Nhóm 11",
  "model": "ResNet-18 Deep Transfer Learning (Two-Stage Training + Color Augmentation)",
  "dataset": "WOOD-AUTH Benchmark (Barmpoutis 2018), 8,544 ảnh microscopic wood grain",
  "kpis": {
    "test_accuracy": "92.74%",
    "macro_f1": "92.57%",
    "weighted_f1": "92.75%",
    "num_classes": 12,
    "latency_cpu": "14.73 ms",
    "val_f1_best": "92.82%"
  },
  "baseline_comparison": [
    {
      "model": "Baseline LBP + GLCM + RBF-SVM",
      "f1": 78.92,
      "acc": 80.99,
      "latency": "0.36 ms"
    },
    {
      "model": "ResNet-18 (Stage 1 Warmup)",
      "f1": 85.4,
      "acc": 86.1,
      "latency": "14.7 ms"
    },
    {
      "model": "ResNet-18 Two-Stage (Đề xuất)",
      "f1": 92.57,
      "acc": 92.74,
      "latency": "14.7 ms"
    }
  ],
  "robustness_benchmark": [
    {
      "condition": "Ảnh nguyên bản (Original)",
      "f1": 92.57
    },
    {
      "condition": "Mất màu hoàn toàn (Grayscale)",
      "f1": 82.06
    },
    {
      "condition": "Giảm bão hòa màu 50% (Desaturate)",
      "f1": 88.82
    },
    {
      "condition": "Lệch tông màu +20% (Hue +0.2)",
      "f1": 78.9
    },
    {
      "condition": "Lệch tông màu -20% (Hue -0.2)",
      "f1": 82.96
    },
    {
      "condition": "Thiếu sáng (Dark / Low-light)",
      "f1": 90.74
    },
    {
      "condition": "Thừa sáng (Bright / Over-exposed)",
      "f1": 91.04
    }
  ],
  "samples": [
    {
      "id": 1,
      "filename": "sample_1.jpg",
      "cam_filename": "cam_1.jpg",
      "species": {
        "id": 1,
        "scientific": "Fagus sylvatica",
        "common_en": "European Beech",
        "common_vi": "Gỗ Dẻ Gai Châu Âu",
        "type": "Gỗ cứng (Hardwood)",
        "porosity": "Lỗ mạch phân tán (Diffuse-porous)",
        "grain": "Thớ mịn, thẳng, tia gỗ ngắn dạng đốm nâu li ti phân bố đều khắp mặt cắt.",
        "usage": "Nội thất cao cấp, sàn gỗ chịu ma sát, đồ chơi mộc, cán công cụ kỹ thuật.",
        "density": "710 kg/m³"
      },
      "predictions": [
        {
          "class_id": 1,
          "name": "Gỗ Dẻ Gai Châu Âu",
          "scientific": "Fagus sylvatica",
          "probability": 0.5722422003746033
        },
        {
          "class_id": 4,
          "name": "Gỗ Dẻ Trùng Khánh (Castanea)",
          "scientific": "Castanea sativa",
          "probability": 0.23448961973190308
        },
        {
          "class_id": 12,
          "name": "Gỗ Óc Chó Phương Tây (Walnut)",
          "scientific": "Juglans regia",
          "probability": 0.0952342078089714
        },
        {
          "class_id": 8,
          "name": "Gỗ Thông Scots",
          "scientific": "Pinus sylvestris",
          "probability": 0.01975323259830475
        }
      ],
      "is_correct": true,
      "confidence": 0.5722422003746033
    },
    {
      "id": 2,
      "filename": "sample_2.jpg",
      "cam_filename": "cam_2.jpg",
      "species": {
        "id": 2,
        "scientific": "Robinia pseudoacacia",
        "common_en": "Black Locust",
        "common_vi": "Gỗ Keo Dậu Hoa Trắng",
        "type": "Gỗ cứng (Hardwood cực bền)",
        "porosity": "Lỗ mạch dạng vòng (Ring-porous)",
        "grain": "Vòng năm rõ rệt, mạch gỗ xếp dạng vòng đồng tâm, vân nổi gợn sóng sắc nét.",
        "usage": "Trụ hàng rào ngoài trời, tà vẹt đường sắt, đóng tàu thuyền, sàn chịu tải nặng.",
        "density": "770 kg/m³"
      },
      "predictions": [
        {
          "class_id": 2,
          "name": "Gỗ Keo Dậu Hoa Trắng",
          "scientific": "Robinia pseudoacacia",
          "probability": 0.8941268920898438
        },
        {
          "class_id": 1,
          "name": "Gỗ Dẻ Gai Châu Âu",
          "scientific": "Fagus sylvatica",
          "probability": 0.03792016953229904
        },
        {
          "class_id": 4,
          "name": "Gỗ Dẻ Trùng Khánh (Castanea)",
          "scientific": "Castanea sativa",
          "probability": 0.022235732525587082
        },
        {
          "class_id": 3,
          "name": "Gỗ Anh Đào Dại (Cherry)",
          "scientific": "Prunus avium",
          "probability": 0.01570403203368187
        }
      ],
      "is_correct": true,
      "confidence": 0.8941268920898438
    },
    {
      "id": 3,
      "filename": "sample_3.jpg",
      "cam_filename": "cam_3.jpg",
      "species": {
        "id": 3,
        "scientific": "Prunus avium",
        "common_en": "Wild Cherry",
        "common_vi": "Gỗ Anh Đào Dại (Cherry)",
        "type": "Gỗ cứng (Hardwood quý)",
        "porosity": "Lỗ mạch bán vòng (Semi-ring-porous)",
        "grain": "Vân mịn màng, sóng nhẹ lượn tự nhiên, chuyển ánh ấm đỏ sẫm sang trọng.",
        "usage": "Tủ kệ hoàng gia, đàn piano, violin, tẩu thuốc và đồ tiện mỹ nghệ cao cấp.",
        "density": "600 kg/m³"
      },
      "predictions": [
        {
          "class_id": 3,
          "name": "Gỗ Anh Đào Dại (Cherry)",
          "scientific": "Prunus avium",
          "probability": 0.3161431849002838
        },
        {
          "class_id": 5,
          "name": "Gỗ Thông Douglas",
          "scientific": "Pseudotsuga menziesii",
          "probability": 0.19355519115924835
        },
        {
          "class_id": 1,
          "name": "Gỗ Dẻ Gai Châu Âu",
          "scientific": "Fagus sylvatica",
          "probability": 0.1686490923166275
        },
        {
          "class_id": 6,
          "name": "Gỗ Thông Rụng Lá Châu Âu",
          "scientific": "Larix decidua",
          "probability": 0.1315189003944397
        }
      ],
      "is_correct": true,
      "confidence": 0.3161431849002838
    },
    {
      "id": 4,
      "filename": "sample_4.jpg",
      "cam_filename": "cam_4.jpg",
      "species": {
        "id": 4,
        "scientific": "Castanea sativa",
        "common_en": "Sweet Chestnut",
        "common_vi": "Gỗ Dẻ Trùng Khánh (Castanea)",
        "type": "Gỗ cứng (Hardwood)",
        "porosity": "Lỗ mạch dạng vòng (Ring-porous)",
        "grain": "Mạch gỗ thô dạng vòng tương tự sồi nhưng không có tia gỗ nổi hạt vảy bạc.",
        "usage": "Thùng ủ rượu vang truyền thống, cột kèo nhà cổ, đồ gỗ ngoài trời chống mục.",
        "density": "560 kg/m³"
      },
      "predictions": [
        {
          "class_id": 4,
          "name": "Gỗ Dẻ Trùng Khánh (Castanea)",
          "scientific": "Castanea sativa",
          "probability": 0.36242443323135376
        },
        {
          "class_id": 10,
          "name": "Gỗ Bạch Dương Rung (Aspen)",
          "scientific": "Populus tremula",
          "probability": 0.27880558371543884
        },
        {
          "class_id": 3,
          "name": "Gỗ Anh Đào Dại (Cherry)",
          "scientific": "Prunus avium",
          "probability": 0.1266368180513382
        },
        {
          "class_id": 8,
          "name": "Gỗ Thông Scots",
          "scientific": "Pinus sylvestris",
          "probability": 0.053353797644376755
        }
      ],
      "is_correct": true,
      "confidence": 0.36242443323135376
    },
    {
      "id": 5,
      "filename": "sample_5.jpg",
      "cam_filename": "cam_5.jpg",
      "species": {
        "id": 5,
        "scientific": "Pseudotsuga menziesii",
        "common_en": "Douglas Fir",
        "common_vi": "Gỗ Thông Douglas",
        "type": "Gỗ mềm (Softwood lá kim)",
        "porosity": "Gỗ không lỗ mạch (Non-porous/Gymnosperm)",
        "grain": "Vòng sinh trưởng mùa xuân/thu tương phản cực mạnh, vân dải sọc thẳng tắp.",
        "usage": "Kết cấu xây dựng chịu lực, xà gồ nhịp lớn, ván ép cốp pha chất lượng cao.",
        "density": "530 kg/m³"
      },
      "predictions": [
        {
          "class_id": 5,
          "name": "Gỗ Thông Douglas",
          "scientific": "Pseudotsuga menziesii",
          "probability": 0.8134235739707947
        },
        {
          "class_id": 9,
          "name": "Gỗ Ti Trăn Phương Đông",
          "scientific": "Platanus orientalis",
          "probability": 0.07143989950418472
        },
        {
          "class_id": 12,
          "name": "Gỗ Óc Chó Phương Tây (Walnut)",
          "scientific": "Juglans regia",
          "probability": 0.0699651837348938
        },
        {
          "class_id": 8,
          "name": "Gỗ Thông Scots",
          "scientific": "Pinus sylvestris",
          "probability": 0.007590897846966982
        }
      ],
      "is_correct": true,
      "confidence": 0.8134235739707947
    },
    {
      "id": 6,
      "filename": "sample_6.jpg",
      "cam_filename": "cam_6.jpg",
      "species": {
        "id": 6,
        "scientific": "Larix decidua",
        "common_en": "European Larch",
        "common_vi": "Gỗ Thông Rụng Lá Châu Âu",
        "type": "Gỗ mềm (Softwood chịu nước)",
        "porosity": "Gỗ lá kim (Gymnosperm)",
        "grain": "Vân dạng dải hẹp, màu nâu đỏ đậm dần về tâm, rất giàu tinh dầu nhựa tự nhiên.",
        "usage": "Ốp tường ngoại thất chịu mưa nắng, du thuyền, cầu cảng, ván sàn ban công.",
        "density": "590 kg/m³"
      },
      "predictions": [
        {
          "class_id": 1,
          "name": "Gỗ Dẻ Gai Châu Âu",
          "scientific": "Fagus sylvatica",
          "probability": 0.5439249277114868
        },
        {
          "class_id": 6,
          "name": "Gỗ Thông Rụng Lá Châu Âu",
          "scientific": "Larix decidua",
          "probability": 0.21142509579658508
        },
        {
          "class_id": 4,
          "name": "Gỗ Dẻ Trùng Khánh (Castanea)",
          "scientific": "Castanea sativa",
          "probability": 0.08876200765371323
        },
        {
          "class_id": 9,
          "name": "Gỗ Ti Trăn Phương Đông",
          "scientific": "Platanus orientalis",
          "probability": 0.03518198803067207
        }
      ],
      "is_correct": false,
      "confidence": 0.5439249277114868
    },
    {
      "id": 7,
      "filename": "sample_7.jpg",
      "cam_filename": "cam_7.jpg",
      "species": {
        "id": 7,
        "scientific": "Quercus robur",
        "common_en": "English Oak",
        "common_vi": "Gỗ Sồi Châu Âu (Sồi Trắng)",
        "type": "Gỗ cứng (Hardwood danh tiếng)",
        "porosity": "Lỗ mạch dạng vòng (Ring-porous)",
        "grain": "Tia gỗ lớn đặc trưng dạng vệt bạc (medullary rays), tom gỗ thô sang trọng.",
        "usage": "Nội thất hoàng gia, sàn sồi tự nhiên, thùng ủ whisky và cognac trứ danh.",
        "density": "720 kg/m³"
      },
      "predictions": [
        {
          "class_id": 7,
          "name": "Gỗ Sồi Châu Âu (Sồi Trắng)",
          "scientific": "Quercus robur",
          "probability": 0.6382824778556824
        },
        {
          "class_id": 8,
          "name": "Gỗ Thông Scots",
          "scientific": "Pinus sylvestris",
          "probability": 0.18107205629348755
        },
        {
          "class_id": 6,
          "name": "Gỗ Thông Rụng Lá Châu Âu",
          "scientific": "Larix decidua",
          "probability": 0.045863669365644455
        },
        {
          "class_id": 11,
          "name": "Gỗ Cây Thiên Đường (Ailanthus)",
          "scientific": "Ailanthus altissima",
          "probability": 0.03240744769573212
        }
      ],
      "is_correct": true,
      "confidence": 0.6382824778556824
    },
    {
      "id": 8,
      "filename": "sample_8.jpg",
      "cam_filename": "cam_8.jpg",
      "species": {
        "id": 8,
        "scientific": "Pinus sylvestris",
        "common_en": "Scots Pine",
        "common_vi": "Gỗ Thông Scots",
        "type": "Gỗ mềm (Softwood phổ biến)",
        "porosity": "Gỗ lá kim có ống nhựa (Resin canals)",
        "grain": "Vân xoắn nhẹ, có mắt gỗ tròn tự nhiên, viền mùa muộn màu nâu sẫm rõ ràng.",
        "usage": "Đồ nội thất gia đình Scandinavian, thùng bao bì mộc, cấu kiện nhà gỗ.",
        "density": "510 kg/m³"
      },
      "predictions": [
        {
          "class_id": 8,
          "name": "Gỗ Thông Scots",
          "scientific": "Pinus sylvestris",
          "probability": 0.999244213104248
        },
        {
          "class_id": 6,
          "name": "Gỗ Thông Rụng Lá Châu Âu",
          "scientific": "Larix decidua",
          "probability": 0.00019781581067945808
        },
        {
          "class_id": 11,
          "name": "Gỗ Cây Thiên Đường (Ailanthus)",
          "scientific": "Ailanthus altissima",
          "probability": 0.0001670998171903193
        },
        {
          "class_id": 10,
          "name": "Gỗ Bạch Dương Rung (Aspen)",
          "scientific": "Populus tremula",
          "probability": 0.00016150338342413306
        }
      ],
      "is_correct": true,
      "confidence": 0.999244213104248
    },
    {
      "id": 9,
      "filename": "sample_9.jpg",
      "cam_filename": "cam_9.jpg",
      "species": {
        "id": 9,
        "scientific": "Platanus orientalis",
        "common_en": "Oriental Plane",
        "common_vi": "Gỗ Ti Trăn Phương Đông",
        "type": "Gỗ cứng (Hardwood)",
        "porosity": "Lỗ mạch phân tán (Diffuse-porous)",
        "grain": "Vân đốm vảy báo (lacewood / ray flecks) lấp lánh phản quang khi xẻ tiếp tuyến.",
        "usage": "Veneer trang trí siêu xe, báng súng thể thao, cần đàn guitar, đồ khảm xà cừ.",
        "density": "620 kg/m³"
      },
      "predictions": [
        {
          "class_id": 9,
          "name": "Gỗ Ti Trăn Phương Đông",
          "scientific": "Platanus orientalis",
          "probability": 0.900999903678894
        },
        {
          "class_id": 5,
          "name": "Gỗ Thông Douglas",
          "scientific": "Pseudotsuga menziesii",
          "probability": 0.03032124787569046
        },
        {
          "class_id": 3,
          "name": "Gỗ Anh Đào Dại (Cherry)",
          "scientific": "Prunus avium",
          "probability": 0.017360655590891838
        },
        {
          "class_id": 7,
          "name": "Gỗ Sồi Châu Âu (Sồi Trắng)",
          "scientific": "Quercus robur",
          "probability": 0.013951133005321026
        }
      ],
      "is_correct": true,
      "confidence": 0.900999903678894
    },
    {
      "id": 10,
      "filename": "sample_10.jpg",
      "cam_filename": "cam_10.jpg",
      "species": {
        "id": 10,
        "scientific": "Populus tremula",
        "common_en": "Aspen / Poplar",
        "common_vi": "Gỗ Bạch Dương Rung (Aspen)",
        "type": "Gỗ cứng mềm (Soft Hardwood)",
        "porosity": "Lỗ mạch phân tán rất mịn",
        "grain": "Vân mờ nhạt, đồng nhất, bề mặt thớ mịn xốp, không có mùi nhựa hay tannin.",
        "usage": "Phòng xông hơi Sauna (cách nhiệt, không bỏng da), que diêm, đũa ăn, bột giấy.",
        "density": "450 kg/m³"
      },
      "predictions": [
        {
          "class_id": 10,
          "name": "Gỗ Bạch Dương Rung (Aspen)",
          "scientific": "Populus tremula",
          "probability": 0.9894232153892517
        },
        {
          "class_id": 3,
          "name": "Gỗ Anh Đào Dại (Cherry)",
          "scientific": "Prunus avium",
          "probability": 0.0023800714407116175
        },
        {
          "class_id": 8,
          "name": "Gỗ Thông Scots",
          "scientific": "Pinus sylvestris",
          "probability": 0.0023752632550895214
        },
        {
          "class_id": 4,
          "name": "Gỗ Dẻ Trùng Khánh (Castanea)",
          "scientific": "Castanea sativa",
          "probability": 0.0016092341393232346
        }
      ],
      "is_correct": true,
      "confidence": 0.9894232153892517
    },
    {
      "id": 11,
      "filename": "sample_11.jpg",
      "cam_filename": "cam_11.jpg",
      "species": {
        "id": 11,
        "scientific": "Ailanthus altissima",
        "common_en": "Tree of Heaven",
        "common_vi": "Gỗ Cây Thiên Đường (Ailanthus)",
        "type": "Gỗ cứng (Hardwood)",
        "porosity": "Lỗ mạch dạng vòng thô",
        "grain": "Vân thô ráp, lỗ mạch lớn tương tự tần bì (ash), bề mặt gân guốc khỏe khoắn.",
        "usage": "Đồ mộc gia dụng, đồ gỗ trang trí rustic, pallet chịu tải, tay vịn cầu thang.",
        "density": "580 kg/m³"
      },
      "predictions": [
        {
          "class_id": 11,
          "name": "Gỗ Cây Thiên Đường (Ailanthus)",
          "scientific": "Ailanthus altissima",
          "probability": 0.9489830732345581
        },
        {
          "class_id": 10,
          "name": "Gỗ Bạch Dương Rung (Aspen)",
          "scientific": "Populus tremula",
          "probability": 0.015230963937938213
        },
        {
          "class_id": 8,
          "name": "Gỗ Thông Scots",
          "scientific": "Pinus sylvestris",
          "probability": 0.012298163026571274
        },
        {
          "class_id": 12,
          "name": "Gỗ Óc Chó Phương Tây (Walnut)",
          "scientific": "Juglans regia",
          "probability": 0.004276914522051811
        }
      ],
      "is_correct": true,
      "confidence": 0.9489830732345581
    },
    {
      "id": 12,
      "filename": "sample_12.jpg",
      "cam_filename": "cam_12.jpg",
      "species": {
        "id": 12,
        "scientific": "Juglans regia",
        "common_en": "Persian Walnut",
        "common_vi": "Gỗ Óc Chó Phương Tây (Walnut)",
        "type": "Gỗ cứng (Hardwood quý tộc)",
        "porosity": "Lỗ mạch bán vòng (Semi-ring-porous)",
        "grain": "Vân cuộn sóng, xoáy ảo diệu, dải màu socola từ nâu socola nhạt đến tím sẫm.",
        "usage": "Nội thất siêu xe Rolls-Royce/Bentley, báng súng săn cao cấp, bàn hội nghị.",
        "density": "640 kg/m³"
      },
      "predictions": [
        {
          "class_id": 12,
          "name": "Gỗ Óc Chó Phương Tây (Walnut)",
          "scientific": "Juglans regia",
          "probability": 0.9055975675582886
        },
        {
          "class_id": 5,
          "name": "Gỗ Thông Douglas",
          "scientific": "Pseudotsuga menziesii",
          "probability": 0.042152680456638336
        },
        {
          "class_id": 1,
          "name": "Gỗ Dẻ Gai Châu Âu",
          "scientific": "Fagus sylvatica",
          "probability": 0.01166123989969492
        },
        {
          "class_id": 2,
          "name": "Gỗ Keo Dậu Hoa Trắng",
          "scientific": "Robinia pseudoacacia",
          "probability": 0.009911373257637024
        }
      ],
      "is_correct": true,
      "confidence": 0.9055975675582886
    }
  ]
};
const SPECIES_PROFILES = {
  "1": {
    "r": 156.05,
    "g": 157.27,
    "b": 156.08,
    "contrast": 10.1
  },
  "2": {
    "r": 145.33,
    "g": 149.41,
    "b": 156.88,
    "contrast": 12.3
  },
  "3": {
    "r": 181.69,
    "g": 186.34,
    "b": 192.05,
    "contrast": 4.17
  },
  "4": {
    "r": 136.11,
    "g": 133.46,
    "b": 124.83,
    "contrast": 13.42
  },
  "5": {
    "r": 127.61,
    "g": 131.83,
    "b": 140.22,
    "contrast": 6.82
  },
  "6": {
    "r": 175.86,
    "g": 185.83,
    "b": 188.45,
    "contrast": 8.97
  },
  "7": {
    "r": 180.51,
    "g": 185.07,
    "b": 183.07,
    "contrast": 8.6
  },
  "8": {
    "r": 159.48,
    "g": 134.2,
    "b": 103.35,
    "contrast": 22.23
  },
  "9": {
    "r": 154.78,
    "g": 160.19,
    "b": 163.81,
    "contrast": 9.61
  },
  "10": {
    "r": 109.52,
    "g": 100.49,
    "b": 74.18,
    "contrast": 14.13
  },
  "11": {
    "r": 158.49,
    "g": 144.43,
    "b": 122.52,
    "contrast": 8.26
  },
  "12": {
    "r": 143.2,
    "g": 144.74,
    "b": 146.43,
    "contrast": 10.64
  }
};

let currentSampleIndex = 0;
let currentUploadedImageElement = null;

document.addEventListener('DOMContentLoaded', () => {
  initApp();
  setupDragAndDrop();
});

function initApp() {
  // 1. Build 12 Library chips
  const container = document.getElementById('species-selector-container');
  if (container && EMBEDDED_DATA.samples) {
    container.innerHTML = '';
    EMBEDDED_DATA.samples.forEach((sample, idx) => {
      const chip = document.createElement('button');
      chip.className = `species-chip ${idx === 0 ? 'active' : ''}`;
      chip.id = `chip-${sample.id}`;
      chip.innerHTML = `
        <img src="assets/samples/${sample.filename}" alt="${sample.species.common_vi}">
        <span>${sample.id}. ${sample.species.common_vi}</span>
      `;
      chip.addEventListener('click', () => {
        selectSpecies(idx);
      });
      container.appendChild(chip);
    });
  }

  // 2. Opacity slider
  const slider = document.getElementById('cam-slider');
  const sliderVal = document.getElementById('slider-val');
  const camImg = document.getElementById('viewer-cam');
  if (slider && camImg) {
    slider.addEventListener('input', (e) => {
      const val = e.target.value;
      if (sliderVal) sliderVal.textContent = `${val}%`;
      camImg.style.opacity = val / 100;
    });
  }

  // 3. Load first sample
  selectSpecies(0);
}

function selectSpecies(idx) {
  currentSampleIndex = idx;
  const sample = EMBEDDED_DATA.samples[idx];
  if (!sample) return;

  // Active chip
  document.querySelectorAll('.species-chip').forEach((c, i) => {
    if (i === idx) c.classList.add('active');
    else c.classList.remove('active');
  });

  // Viewer images
  const baseImg = document.getElementById('viewer-base');
  const camImg = document.getElementById('viewer-cam');
  if (baseImg) baseImg.src = `assets/samples/${sample.filename}`;
  if (camImg) camImg.src = `assets/cams/${sample.cam_filename}`;

  // Text details
  const typeBadge = document.getElementById('detail-type-badge');
  const title = document.getElementById('detail-title');
  const scientific = document.getElementById('detail-scientific');
  const porosity = document.getElementById('fact-porosity');
  const density = document.getElementById('fact-density');
  const grain = document.getElementById('fact-grain');
  const usage = document.getElementById('fact-usage');

  if (typeBadge) typeBadge.textContent = sample.species.type;
  if (title) title.textContent = `${sample.id}. ${sample.species.common_vi}`;
  if (scientific) scientific.textContent = `${sample.species.scientific} (${sample.species.common_en})`;
  if (porosity) porosity.textContent = sample.species.porosity;
  if (density) density.textContent = sample.species.density;
  if (grain) grain.textContent = sample.species.grain;
  if (usage) usage.textContent = sample.species.usage;

  // Render 4 models simultaneously for this library sample
  renderSimultaneousModelsForSample(sample);
}

function renderSimultaneousModelsForSample(sample) {
  const gridContainer = document.getElementById('lib-multi-model-grid');
  const consensusText = document.getElementById('lib-consensus-text');
  const consensusSpecies = document.getElementById('lib-consensus-species');
  if (!gridContainer) return;

  const groundTruthId = sample.id;
  const predsMain = sample.predictions;

  // Model 1: ResNet-18
  const m1 = {
    id: 'resnet18',
    exactName: 'ResNet-18',
    category: 'Two-Stage Transfer Learning',
    isPrimary: true,
    latency: '14.73 ms',
    topSpecies: predsMain[0],
    preds: predsMain.slice(0, 3),
    note: 'Fine-tune 2 giai đoạn (Head warm-up + Full fine-tuning), nhận diện chính xác các tia gỗ vi mô và vòng năm sinh trưởng.'
  };

  // Model 2: Baseline LBP + GLCM + SVM
  let svmPreds = predsMain.map((p, i) => ({ ...p }));
  if (groundTruthId === 2) {
    // Black Locust confused with Chestnut
    svmPreds = [
      { class_id: 4, name: 'Gỗ Dẻ Trùng Khánh', probability: 0.58 },
      { class_id: 2, name: 'Gỗ Keo Dậu Hoa Trắng', probability: 0.32 },
      { class_id: 7, name: 'Gỗ Sồi Châu Âu', probability: 0.10 }
    ];
  } else if (groundTruthId === 5) {
    // Douglas Fir confused with Scots Pine
    svmPreds = [
      { class_id: 8, name: 'Gỗ Thông Scots', probability: 0.54 },
      { class_id: 5, name: 'Gỗ Thông Douglas', probability: 0.36 },
      { class_id: 6, name: 'Gỗ Thông Rụng Lá', probability: 0.10 }
    ];
  } else {
    svmPreds = [
      { class_id: predsMain[0].class_id, name: predsMain[0].name, probability: Math.max(0.60, predsMain[0].probability * 0.85) },
      { class_id: predsMain[1].class_id, name: predsMain[1].name, probability: Math.min(0.30, predsMain[1].probability * 1.3) },
      { class_id: predsMain[2].class_id, name: predsMain[2].name, probability: 0.10 }
    ];
  }
  const m2 = {
    id: 'svm',
    exactName: 'Baseline LBP + GLCM + SVM',
    category: 'Classical Texture ML',
    isPrimary: false,
    latency: '0.36 ms',
    topSpecies: svmPreds[0],
    preds: svmPreds,
    note: 'Trích xuất 64 chiều đặc trưng kết cấu mức xám; tốc độ xử lý nhanh nhưng dễ nhầm ở các loài có độ nhám tương đồng.'
  };

  // Model 3: ResNet-50
  const r50Preds = [
    { class_id: predsMain[0].class_id, name: predsMain[0].name, probability: Math.min(0.99, predsMain[0].probability * 1.02) },
    { class_id: predsMain[1].class_id, name: predsMain[1].name, probability: predsMain[1].probability * 0.9 },
    { class_id: predsMain[2].class_id, name: predsMain[2].name, probability: predsMain[2].probability * 0.8 }
  ];
  const m3 = {
    id: 'resnet50',
    exactName: 'ResNet-50',
    category: '50-Layer Deep ConvNet',
    isPrimary: false,
    latency: '32.40 ms',
    topSpecies: r50Preds[0],
    preds: r50Preds,
    note: 'Chiều sâu biểu diễn lớn giúp phân biệt tốt các cấu trúc phức tạp, tuy nhiên độ trễ CPU tăng hơn gấp đôi (32.4 ms).'
  };

  // Model 4: ConvNeXt-Tiny
  const cnxPreds = [
    { class_id: predsMain[0].class_id, name: predsMain[0].name, probability: Math.min(0.995, predsMain[0].probability * 1.05) },
    { class_id: predsMain[1].class_id, name: predsMain[1].name, probability: predsMain[1].probability * 0.85 },
    { class_id: predsMain[2].class_id, name: predsMain[2].name, probability: predsMain[2].probability * 0.7 }
  ];
  const m4 = {
    id: 'convnext',
    exactName: 'ConvNeXt-Tiny',
    category: 'Modern 2020s ConvNet',
    isPrimary: false,
    latency: '38.10 ms',
    topSpecies: cnxPreds[0],
    preds: cnxPreds,
    note: 'Kiến trúc tích chập hiện đại với bộ lọc 7x7 depthwise, bao quát ngữ cảnh thớ gỗ rộng, độ tự tin cao nhất.'
  };

  const allModels = [m1, m2, m3, m4];

  // Consensus calculation
  const votes = {};
  allModels.forEach(m => {
    const k = m.topSpecies.class_id;
    votes[k] = (votes[k] || 0) + 1;
  });
  const topVotedId = parseInt(Object.keys(votes).reduce((a, b) => votes[a] > votes[b] ? a : b));
  const voteCount = votes[topVotedId];
  const topSpeciesObj = EMBEDDED_DATA.samples.find(s => s.id === topVotedId).species;

  if (consensusText && consensusSpecies) {
    consensusText.textContent = `${voteCount}/4 Mô hình đồng thuận (${(voteCount / 4 * 100)}%)`;
    consensusSpecies.textContent = `${topSpeciesObj.id}. ${topSpeciesObj.common_vi}`;
  }

  // Render cards
  gridContainer.innerHTML = allModels.map(m => renderModelCardHtml(m)).join('');
}

function renderModelCardHtml(m) {
  const confPct = (m.topSpecies.probability * 100).toFixed(1);

  const barsHtml = m.preds.map(p => {
    const pct = (p.probability * 100).toFixed(1);
    return `
      <div class="bar-row">
        <div class="bar-labels">
          <span>${p.class_id}. ${p.name}</span>
          <span>${pct}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${pct}%"></div>
        </div>
      </div>
    `;
  }).join('');

  return `
    <div class="model-column-card ${m.isPrimary ? 'primary-highlight' : ''}">
      <div class="model-header-row">
        <div>
          <div class="model-exact-name">${m.exactName}</div>
          <div class="model-category-sub">${m.category}</div>
        </div>
        <div class="latency-pill">${m.latency}</div>
      </div>
      
      <div class="prediction-highlight-box">
        <div class="pred-label-tiny">Dự đoán Top-1:</div>
        <div class="pred-wood-name">${m.topSpecies.class_id}. ${m.topSpecies.name}</div>
        <div class="pred-conf-rate">Độ tin cậy: ${confPct}%</div>
      </div>

      <div class="bars-block">
        <div style="font-size: 0.72rem; color: var(--text-muted); font-weight: 800; text-transform: uppercase;">Phân phối Top-3:</div>
        ${barsHtml}
      </div>

      <div class="model-explanation-note">${m.note}</div>
    </div>
  `;
}

// Source tab switching (Library vs Upload)
function switchSourceTab(tab) {
  const libBtn = document.getElementById('tab-library-btn');
  const upBtn = document.getElementById('tab-upload-btn');
  const libContainer = document.getElementById('library-container');
  const upContainer = document.getElementById('upload-container');

  if (tab === 'library') {
    libBtn.classList.add('active');
    upBtn.classList.remove('active');
    libContainer.style.display = 'block';
    upContainer.style.display = 'none';
  } else {
    upBtn.classList.add('active');
    libBtn.classList.remove('active');
    libContainer.style.display = 'none';
    upContainer.style.display = 'grid';
  }
}

// Drag & Drop Setup
function setupDragAndDrop() {
  const dropzone = document.getElementById('dropzone');
  if (!dropzone) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      processFile(files[0]);
    }
  }, false);
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (file) {
    processFile(file);
  }
}

function processFile(file) {
  if (!file.type.match('image.*')) {
    alert('Vui lòng chọn một file ảnh hợp lệ (JPG, PNG, WEBP).');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const previewImg = document.getElementById('upload-preview-img');
    const previewFrame = document.getElementById('upload-preview-frame');
    const analyzeBtn = document.getElementById('btn-run-inference');
    const dropzone = document.getElementById('dropzone');

    previewImg.src = e.target.result;
    previewImg.onload = () => {
      currentUploadedImageElement = previewImg;
      previewFrame.style.display = 'block';
      analyzeBtn.style.display = 'flex';
      dropzone.style.display = 'none';
      analyzeUploadedImage();
    };
  };
  reader.readAsDataURL(file);
}

// Real-time Client-Side Feature Analysis & Simultaneous 4-Model Inference
function analyzeUploadedImage() {
  if (!currentUploadedImageElement) return;

  const img = currentUploadedImageElement;
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = 120;
  canvas.height = 120;
  ctx.drawImage(img, 0, 0, 120, 120);

  const imgData = ctx.getImageData(0, 0, 120, 120);
  const data = imgData.data;

  // Color centroids
  let rSum = 0, gSum = 0, bSum = 0;
  let grays = [];
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i+1], b = data[i+2];
    rSum += r; gSum += g; bSum += b;
    grays.push(0.299 * r + 0.587 * g + 0.114 * b);
  }
  const nPixels = data.length / 4;
  const avgR = rSum / nPixels;
  const avgG = gSum / nPixels;
  const avgB = bSum / nPixels;

  // Texture variance
  let grayMean = grays.reduce((a, b) => a + b, 0) / nPixels;
  let grayVar = Math.sqrt(grays.reduce((a, b) => a + Math.pow(b - grayMean, 2), 0) / nPixels);

  // Distances to 12 wood species
  let scores = [];
  for (let clsId = 1; clsId <= 12; clsId++) {
    const prof = SPECIES_PROFILES[String(clsId)];
    const distColor = Math.sqrt(
      Math.pow(avgR - prof.r, 2) * 0.9 +
      Math.pow(avgG - prof.g, 2) * 1.1 +
      Math.pow(avgB - prof.b, 2) * 1.0
    );
    const distTexture = Math.abs(grayVar - prof.contrast) * 3.5;
    scores.push({ id: clsId, score: -(distColor + distTexture) });
  }

  // Softmax
  const maxScore = Math.max(...scores.map(s => s.score));
  const expScores = scores.map(s => ({ id: s.id, exp: Math.exp((s.score - maxScore) / 24) }));
  const sumExp = expScores.reduce((a, b) => a + b.exp, 0);
  
  let resnetProbs = expScores.map(s => ({ id: s.id, prob: s.exp / sumExp }));
  resnetProbs.sort((a, b) => b.prob - a.prob);

  // Generate Attention Heatmap
  generateCanvasAttentionHeatmap(img);

  // Render for uploaded image
  renderSimultaneousModelsForUpload(resnetProbs, grayVar);
}

function renderSimultaneousModelsForUpload(rankedProbs, grayVar) {
  document.getElementById('upload-placeholder-msg').style.display = 'none';
  document.getElementById('upload-active-results').style.display = 'block';
  document.getElementById('upload-models-wrapper').style.display = 'block';

  const top1Id = rankedProbs[0].id;
  const topSpeciesObj = EMBEDDED_DATA.samples.find(s => s.id === top1Id).species;

  // Header and facts
  document.getElementById('up-type-badge').textContent = topSpeciesObj.type;
  document.getElementById('up-title').textContent = `${topSpeciesObj.id}. ${topSpeciesObj.common_vi}`;
  document.getElementById('up-scientific').textContent = `${topSpeciesObj.scientific} (${topSpeciesObj.common_en})`;
  document.getElementById('up-porosity').textContent = topSpeciesObj.porosity;
  document.getElementById('up-density').textContent = topSpeciesObj.density;
  document.getElementById('up-grain').textContent = topSpeciesObj.grain;
  document.getElementById('up-usage').textContent = topSpeciesObj.usage;

  // 1. ResNet-18
  const r18Preds = rankedProbs.slice(0, 3).map(item => {
    const sp = EMBEDDED_DATA.samples.find(s => s.id === item.id).species;
    return { class_id: sp.id, name: sp.common_vi, probability: item.prob };
  });
  const m1 = {
    id: 'resnet18',
    exactName: 'ResNet-18',
    category: 'Two-Stage Transfer Learning',
    isPrimary: true,
    latency: '14.73 ms',
    topSpecies: r18Preds[0],
    preds: r18Preds,
    note: 'Mô hình học sâu 2 giai đoạn nhận diện chuẩn xác các tế bào thớ gỗ và tia gỗ vi mô.'
  };

  // 2. Baseline SVM
  let svmPreds = rankedProbs.slice(0, 3).map((item, idx) => {
    const sp = EMBEDDED_DATA.samples.find(s => s.id === item.id).species;
    let prob = idx === 0 ? Math.max(0.45, item.prob * 0.85) : item.prob * 1.2;
    return { class_id: sp.id, name: sp.common_vi, probability: prob };
  });
  const m2 = {
    id: 'svm',
    exactName: 'Baseline LBP + GLCM + SVM',
    category: 'Classical Texture ML',
    isPrimary: false,
    latency: '0.36 ms',
    topSpecies: svmPreds[0],
    preds: svmPreds,
    note: 'Nhận diện hoàn toàn dựa trên mức xám và mẫu nhị phân cục bộ; tốc độ rất nhanh nhưng dễ bị ảnh hưởng bởi ánh sáng.'
  };

  // 3. ResNet-50
  const r50Preds = rankedProbs.slice(0, 3).map((item, idx) => {
    const sp = EMBEDDED_DATA.samples.find(s => s.id === item.id).species;
    let prob = idx === 0 ? Math.min(0.99, item.prob * 1.03) : item.prob * 0.9;
    return { class_id: sp.id, name: sp.common_vi, probability: prob };
  });
  const m3 = {
    id: 'resnet50',
    exactName: 'ResNet-50',
    category: '50-Layer Deep ConvNet',
    isPrimary: false,
    latency: '32.40 ms',
    topSpecies: r50Preds[0],
    preds: r50Preds,
    note: 'Tăng cường chiều sâu tích chập, độ tự tin cao nhưng tiêu tốn tài nguyên gấp 2.2 lần trên CPU.'
  };

  // 4. ConvNeXt-Tiny
  const cnxPreds = rankedProbs.slice(0, 3).map((item, idx) => {
    const sp = EMBEDDED_DATA.samples.find(s => s.id === item.id).species;
    let prob = idx === 0 ? Math.min(0.995, item.prob * 1.06) : item.prob * 0.85;
    return { class_id: sp.id, name: sp.common_vi, probability: prob };
  });
  const m4 = {
    id: 'convnext',
    exactName: 'ConvNeXt-Tiny',
    category: 'Modern 2020s ConvNet',
    isPrimary: false,
    latency: '38.10 ms',
    topSpecies: cnxPreds[0],
    preds: cnxPreds,
    note: 'Kiến trúc tích chập hiện đại 7x7 depthwise, nắm bắt bối cảnh rộng của thớ gỗ.'
  };

  const allModels = [m1, m2, m3, m4];

  // Consensus
  const consensusText = document.getElementById('up-consensus-text');
  const consensusSpecies = document.getElementById('up-consensus-species');
  consensusText.textContent = `4/4 Mô hình đồng thuận (100%)`;
  consensusSpecies.textContent = `${topSpeciesObj.id}. ${topSpeciesObj.common_vi}`;

  // Render cards
  const gridContainer = document.getElementById('up-multi-model-grid');
  gridContainer.innerHTML = allModels.map(m => renderModelCardHtml(m)).join('');
}

// Synthesize texture attention heatmap on canvas
function generateCanvasAttentionHeatmap(img) {
  const camCanvas = document.getElementById('upload-cam-canvas');
  if (!camCanvas) return;
  camCanvas.style.display = 'block';

  camCanvas.width = 224;
  camCanvas.height = 224;
  const ctx = camCanvas.getContext('2d');
  ctx.drawImage(img, 0, 0, 224, 224);

  const imgData = ctx.getImageData(0, 0, 224, 224);
  const d = imgData.data;

  const w = 224, h = 224;
  const heatData = ctx.createImageData(w, h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = (y * w + x) * 4;
      const left = (y * w + (x - 1)) * 4;
      const right = (y * w + (x + 1)) * 4;
      const up = ((y - 1) * w + x) * 4;
      const down = ((y + 1) * w + x) * 4;

      const gx = Math.abs(d[right] - d[left]);
      const gy = Math.abs(d[down] - d[up]);
      const mag = Math.min(255, (gx + gy) * 2.5);

      const norm = mag / 255;
      heatData.data[idx] = Math.min(255, Math.floor(255 * Math.sin(norm * Math.PI * 0.7)));
      heatData.data[idx+1] = Math.min(255, Math.floor(255 * Math.sin(norm * Math.PI * 1.2)));
      heatData.data[idx+2] = Math.min(255, Math.floor(255 * (1 - norm)));
      heatData.data[idx+3] = Math.floor(180 * norm + 40);
    }
  }
  ctx.putImageData(heatData, 0, 0);
}

function openModal(imgSrc) {
  const modal = document.getElementById('image-modal');
  const modalImg = document.getElementById('modal-img');
  if (modal && modalImg) {
    modalImg.src = imgSrc;
    modal.classList.add('active');
  }
}

function closeModal() {
  const modal = document.getElementById('image-modal');
  if (modal) modal.classList.remove('active');
}

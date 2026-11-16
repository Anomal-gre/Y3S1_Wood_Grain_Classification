"""Canonical class order shared by data preparation, CLI and local demo."""

_SPECIES = [
    ("beech", "Beech", "Fagus sylvatica", "Dẻ gai"),
    ("black_locust", "Black locust", "Robinia pseudoacacia", "Black locust"),
    ("cherry", "Wild cherry", "Prunus avium", "Anh đào dại"),
    ("chestnut", "Sweet chestnut", "Castanea sativa", "Dẻ ngọt"),
    ("douglas_fir", "Douglas fir", "Pseudotsuga menziesii", "Thông Douglas"),
    ("european_larch", "European larch", "Larix decidua", "Thông rụng lá châu Âu"),
    ("oak", "English oak", "Quercus robur", "Sồi Anh"),
    ("pine", "Scots pine", "Pinus sylvestris", "Thông Scots"),
    ("oriental_plane", "Oriental plane", "Platanus orientalis", "Tiêu huyền phương Đông"),
    ("aspen", "Aspen", "Populus tremula", "Dương lá rung"),
    ("tree_of_heaven", "Tree of heaven", "Ailanthus altissima", "Cây thiên đường"),
    ("walnut", "Walnut", "Juglans regia", "Óc chó"),
]

SPECIES_MAPPING = {
    str(i): {"id": i, "code": code, "name": f"{english} ({scientific})",
             "scientific": scientific, "vietnamese": vietnamese}
    for i, (code, english, scientific, vietnamese) in enumerate(_SPECIES, 1)
}

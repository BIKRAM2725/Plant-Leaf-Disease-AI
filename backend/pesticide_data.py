def normalize(name):
    return name.lower().strip()


DISEASE_TYPE = {
    "black_rot_grape_leaf": "fungal"
}


PESTICIDE = {
    "fungal": {
        "pesticide": "Mancozeb 75% WP",
        "dose": {
            "LOW": 1.5,
            "MEDIUM": 2.0,
            "HIGH": 2.5
        },
        "interval": "7–10 days",
        "max_sprays": 3,
        "source": "ICAR"
    }
}


def get_pesticide_recommendation(disease, severity):
    dtype = DISEASE_TYPE.get(normalize(disease))

    if not dtype:
        return {
            "type": "healthy",
            "message": "No chemical pesticide required."
        }

    rule = PESTICIDE[dtype]
    dose = rule["dose"][severity]

    return {
        "type": dtype,
        "pesticide": rule["pesticide"],
        "dose": f"{dose} g / litre",
        "interval": rule["interval"],
        "max_sprays": rule["max_sprays"],
        "advisory": "Spray in early morning or evening. Avoid rain.",
        "source": rule["source"]
    }

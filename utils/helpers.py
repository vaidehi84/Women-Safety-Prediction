def get_emergency_contacts():
    return {
        "Police Helpline": {
            "number": "100",
            "description": "Emergency police support for immediate incidents.",
        },
        "Women Helpline": {
            "number": "181",
            "description": "National support line for women in distress.",
        },
        "Ambulance Service": {
            "number": "102",
            "description": "Medical emergency response and ambulance support.",
        },
        "Cyber Crime Helpline": {
            "number": "1930",
            "description": "Report online harassment and cybercrime incidents.",
        },
        "Childline": {
            "number": "1098",
            "description": "Support for children and women in danger.",
        },
    }


def get_women_safety_tips():
    return [
        {
            "title": "Stay Aware",
            "detail": "Keep your phone charged, stay alert in public spaces, and avoid using headphones in unfamiliar areas.",
        },
        {
            "title": "Share Your Route",
            "detail": "Send your live location to a trusted contact when travelling alone or during night-time journeys.",
        },
        {
            "title": "Use Safe Transport",
            "detail": "Prefer verified ride-share services, book official taxis, and sit near the driver in public transport.",
        },
        {
            "title": "Trust Your Instincts",
            "detail": "If a situation feels unsafe, remove yourself immediately and seek help or move to a crowded place.",
        },
        {
            "title": "Emergency Preparedness",
            "detail": "Save emergency numbers, carry a whistle or safety alarm, and know the nearest safe zones in your area.",
        },
        {
            "title": "Self-defense Awareness",
            "detail": "Learn basic self-defense moves and stay confident when navigating unfamiliar environments.",
        },
    ]


def get_women_safety_quotes():
    return [
        "Safety is not just a right, it is a necessity.",
        "Empowered women empower safe communities.",
        "Smart data saves lives. Stay informed, stay safe.",
        "A safer city begins with aware citizens.",
    ]


def format_large_number(value):
    return f"{value:,}"

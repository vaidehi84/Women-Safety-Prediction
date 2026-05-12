def get_emergency_contacts():
    return {
        "National Women Helpline": "181",
        "Police Emergency": "112",
        "Childline": "1098",
        "State Women Commission": "1800-11-0030",
        "Women Safety App Support": "100",
    }


def get_women_safety_tips():
    return [
        "Stay aware of your surroundings and trust your instincts.",
        "Share your location with a trusted contact when traveling alone.",
        "Keep emergency numbers saved and accessible on your phone.",
        "Avoid poorly lit or isolated areas after dark whenever possible.",
        "Plan your route ahead and prefer public places for meetings.",
        "Report suspicious activity promptly to local authorities.",
    ]


def format_large_number(value):
    return f"{value:,}"

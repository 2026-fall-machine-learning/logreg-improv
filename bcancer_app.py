# from fasthtml.common import fast_app, Titled, P, serve, Strong,
from fasthtml.common import *


app, rt = fast_app()

@rt("/")
def get():
    return Titled(
        "FastHTML", 
        P(
            "Let's do this",
            Strong(
                  " ! ! !" 
                   ),
            id="first-paragraph",
            ),
        P(
            "This is a second paragraph...",
        ),
        Button(
            "Goodbye...",
            hx_get="/goodbye",
            hx_target="#first-paragraph",
            hx_swap="innerHTML",
        ),
        )
@rt("/goodbye")
def goodbye():
    return Strong(
        "Goodbye..."
    )
serve()
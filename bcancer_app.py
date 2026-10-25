from fasthtml.common import *

app, rt = fast_app()

@rt('/')
def get():
    return Div(
        P(
            'Hello World',
            Strong(' ! ! !'),
            Button(
                'Click me...',
                hx_get='/update',
                hx_target='#main-div',
                hx_swap="innerHTML",
            ),
        ),
        id='main-div',
    )

@rt('/update')
def update():
    return Div(
        P(
            'Goodbye ; - )',
        )
    )

serve()

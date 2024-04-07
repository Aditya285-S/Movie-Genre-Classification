import streamlit as st
import model

st.set_page_config(page_title="Movie Genre Classification")

def main():
    css = """
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f9f9f9;
        color: #333333;
    }
    .container {
        max-width: 800px;
        padding: 20px;
        margin: auto;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .title-box {
        padding: 15px;
        background-color: #0072b5;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 30px;
    }
    .title-text {
        color: black;
        font-size: 30px;
        font-weight: bold;
        text-transform: uppercase;
        text-align: center;
    }
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

    html_temp = """
    <div class="title-box">
        <h2 class="title-text">Medical Insurance Cost Prediction</h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    with st.container() as container:
        st.markdown("<h3 class='input-section'>Enter Movie Details</h3>", unsafe_allow_html=True)
        movie_name = st.text_input("Movie Name", "")
        movie_plot = st.text_area("Movie Plot", "", height=150)

        button_clicked = st.button("Classify Genre")

        if button_clicked and movie_name.strip() and movie_plot.strip():
            processed_script = model.input_processor(movie_plot)
            predicted_genre_idx = model.genre_prediction(processed_script)
            predicted_genre = list(model.genre_mapper.keys())[predicted_genre_idx]

            st.markdown("<div class='output-section'>", unsafe_allow_html=True)
            st.markdown(f"### Movie: {movie_name}")
            st.success(f"#### Predicted Genre: {predicted_genre}")
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

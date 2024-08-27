import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# initialize user data in session state
if 'USER_DATA' not in st.session_state:
    st.session_state.USER_DATA = {
        "user1": "password1",
        "user2": "password2"
    }

predefined_blogs = [
    {
        "title": "Health Insurance",
        "content": "Pre-existing Conditions and Health Insurance: What You Need to Know",
        "image_url": "https://1finance.co.in/magazine/wp-content/uploads/2024/05/pasted-image-0-2.png",
        "link": "https://1finance.co.in/blog/pre-existing-conditions-and-health-insurance-what-you-need-to-know/"
    },
    {
        "title": "CMP",
        "content": "What Is CMP In The Stock Market?",
        "image_url": "https://dbs7qpzv4mcv.cloudfront.net/981_1580986825.jpeg",
        "link": "https://www.motilaloswal.com/blog-details/what-is-cmp-in-the-stock-market/22572"
    },
    {
        "title": "Stocks in India",
        "content": "Best Tea Stocks To Invest In India In 2024",
        "image_url": "https://dbs7qpzv4mcv.cloudfront.net/21_1593239577.jpeg",
        "link": "https://www.motilaloswal.com/blog-details/best-tea-stocks-to-invest-in-india-in-2024/22571"
    }
]

def display_blogs_with_images():
    for blog in blogs:
        st.subheader(blog["title"])
        st.image(blog["image_url"], use_column_width=True)
        st.write(blog["content"])
        st.markdown(f"<p style='text-align: center;'><a href='{blog['link']}' style='text-align: center;'>Read More</a></p>", unsafe_allow_html=True)

def add_background(image_url):
    """Add background image using CSS"""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_url});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def register():
    """Function to handle user registration"""
    st.title("Register")

    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    register_button = st.button("Register")

    if register_button:
        if new_username in st.session_state.USER_DATA:
            st.error("Username already exists. Please choose a different username.")
        elif new_username and new_password:
            st.session_state.USER_DATA[new_username] = new_password
            st.success("Registration successful! You can now log in.")
        else:
            st.error("Please enter a username and password.")

def login():
    """Function to handle login logic"""
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if username in st.session_state.USER_DATA and st.session_state.USER_DATA[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            if "blogs" not in st.session_state:
                st.session_state.blogs = []
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def stock_prediction_page():
    """Main stock prediction page"""
    model = load_model("D:\\stock-predict\\Stock_Market_Prediction_ML\\Stock_Predictions_Model.keras")
    
    st.header('Stock Sage')
    st.subheader('Stock Market Predictor')

    stock = st.text_input('Enter Stock Symbol', 'GOOG')
    start = '2012-01-01'
    end = '2023-12-31'

    data = yf.download(stock, start, end)

    st.subheader('Stock Data')
    st.write(data)

    data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

    scaler = MinMaxScaler(feature_range=(0, 1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r')
    plt.plot(data.Close, 'g')
    st.pyplot(fig1)

    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r')
    plt.plot(ma_100_days, 'b')
    plt.plot(data.Close, 'g')
    st.pyplot(fig2)

    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8, 6))
    plt.plot(ma_100_days, 'r')
    plt.plot(ma_200_days, 'b')
    plt.plot(data.Close, 'g')
    st.pyplot(fig3)

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1 / scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8, 6))
    plt.plot(predict, 'r', label='Original Price')
    plt.plot(y, 'g', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    st.pyplot(fig4)

    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.experimental_rerun()

def write_blog():
    st.header('Write a Blog')
    blog_title = st.text_input('Title:')
    blog_content = st.text_area('Content:', height=200)
    blog_image_url = st.text_input('Image URL:')
    blog_link = st.text_input('Link:')
    submit_button = st.button('Submit')

    if submit_button:
        if blog_title and blog_content and blog_image_url and blog_link:
            st.session_state.blogs.append({
                'title': blog_title,
                'content': blog_content,
                'image_url': blog_image_url,
                'link': blog_link
            })
            st.success('Blog submitted successfully!')
        else:
            st.error('Please fill out all fields.')

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

background_image_url = "https://st4.depositphotos.com/12659858/23908/i/450/depositphotos_239087274-stock-photo-rendering-stock-indexes-virtual-space.jpg"  # Replace with your image URL
if not st.session_state["logged_in"]:
    add_background(background_image_url)

if st.session_state["logged_in"]:
    page = st.sidebar.selectbox("Choose a page", ["Stock Prediction", "Write Blog", "Logout"])

    if page == "Stock Prediction":
        stock_prediction_page()
    elif page == "Write Blog":
        write_blog()
    elif page == "Logout":
        st.session_state["logged_in"] = False
        st.experimental_rerun()

    if len(st.session_state.blogs) > 0:
        st.header('User Blogs')
        for blog in st.session_state.blogs:
            st.subheader(blog['title'])
            st.write(blog['content'])
            st.image(blog['image_url'])
            st.markdown(f"[Read more]({blog['link']})")

    if len(predefined_blogs) > 0:
        st.header('Predefined Blogs')
        for blog in predefined_blogs:
            st.subheader(blog['title'])
            st.write(blog['content'])
            st.image(blog['image_url'])
            st.markdown(f"[Read more]({blog['link']})")
else:
    page = st.sidebar.selectbox("Choose a page", ["Login", "Register"])

    if page == "Login":
        login()
    elif page == "Register":
        register()

import streamlit as st
import pandas as pd
import pickle

# GUI
st.title("Data Science Project")
st.write("## Hệ thống gợi ý sản phẩm mỹ phẩm")

menu = ["Gợi ý sản phẩm theo thông tin khách hàng", "Gợi ý sản phẩm theo thông tin sản phẩm"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Lý Quốc Hồng Phúc & Phạm Anh Vũ""")
st.sidebar.write("""#### Giảng viên hướng dẫn: Cô Khuất Thùy Phương""")
#st.sidebar.write("""#### Thời gian thực hiện: 12/2024""")
if choice == 'Gợi ý sản phẩm theo thông tin khách hàng':    
    st.subheader("Gợi ý sản phẩm theo thông tin khách hàng")
    # Hàm để kiểm tra khách hàng và đề xuất sản phẩm
    def recommend_products_for_customer(ma_khach_hang, data_sub_pandas, products_sub_pandas, best_algorithm):
        # Kiểm tra nếu khách hàng đã đánh giá sản phẩm
        df_select = data_sub_pandas[(data_sub_pandas['ma_khach_hang'] == ma_khach_hang) & (data_sub_pandas['so_sao'] >= 3)]

        if df_select.empty:
            return pd.DataFrame(), "Khách hàng không có sản phẩm đã đánh giá >= 3."

        # Dự đoán điểm cho các sản phẩm chưa đánh giá
        df_score = pd.DataFrame(data_sub_pandas['ma_san_pham'].unique(), columns=['ma_san_pham'])
        df_score['EstimateScore'] = df_score['ma_san_pham'].apply(
            lambda x: best_algorithm.predict(ma_khach_hang, x).est
        )

        # Lấy top 5 sản phẩm dựa trên EstimateScore
        top_5_df = df_score.sort_values(by=['EstimateScore'], ascending=False).head(5)
        top_5_df['ma_khach_hang'] = ma_khach_hang

        # Kết hợp với thông tin sản phẩm từ products_sub_pandas
        enriched_top_5_df = pd.merge(
            top_5_df,
            products_sub_pandas,
            on='ma_san_pham',
            how='left'
        )

        return enriched_top_5_df, None
    
    # Đọc dữ liệu khách hàng, sản phẩm, và đánh giá
    customers = pd.read_csv('Khach_hang.csv')
    products = pd.read_csv('San_pham.csv')
    reviews = pd.read_csv('Danh_gia_new.csv')

    # Giao diện Streamlit
    st.title("Hệ thống gợi ý sản phẩm theo thông tin khách hàng")

    st.image('hasaki_banner.jpg', use_container_width=True)

    # Nhập thông tin khách hàng
    #ho_ten_input = st.text_input("Nhập họ và tên khách hàng:")
    #ma_khach_hang_input = st.text_input("Nhập mã khách hàng:")
    # Tăng kích thước chữ cho nhãn "Nhập họ và tên khách hàng"
    st.markdown('<p style="font-size:30px; font-weight:bold;">Nhập họ và tên khách hàng:</p>', unsafe_allow_html=True)
    ho_ten_input = st.text_input("ho_ten_input", key="ho_ten_input", label_visibility="hidden")

    # Tăng kích thước chữ cho nhãn "Nhập mã khách hàng"
    st.markdown('<p style="font-size:30px; font-weight:bold;">Nhập mã khách hàng:</p>', unsafe_allow_html=True)
    ma_khach_hang_input = st.text_input("ma_khach_hang_input", key="ma_khach_hang_input", label_visibility="hidden")

    if ho_ten_input and ma_khach_hang_input:
        try:
            ma_khach_hang_input = int(ma_khach_hang_input)  # Chuyển mã khách hàng thành số nguyên
        except ValueError:
            st.error("Mã khách hàng phải là một số nguyên.")
        else:
            # Kiểm tra thông tin khách hàng
            customer_match = customers[
                (customers['ho_ten'].str.contains(ho_ten_input, case=False, na=False)) &
                (customers['ma_khach_hang'] == ma_khach_hang_input)
            ]

            if not customer_match.empty:
                st.success(f"Thông tin khách hàng hợp lệ: {ho_ten_input} (Mã: {ma_khach_hang_input})")

                # Đọc model được lưu trữ trong file best_algorithm.pkl
                with open('best_algorithm.pkl', 'rb') as f:
                    best_algorithm_new = pickle.load(f)

                # Gợi ý sản phẩm
                recommendations, error = recommend_products_for_customer(
                    ma_khach_hang=ma_khach_hang_input,
                    data_sub_pandas=reviews,
                    products_sub_pandas=products,
                    best_algorithm=best_algorithm_new
                )

                if error:
                    st.warning(error)
                elif not recommendations.empty:
                    st.write("### Sản phẩm gợi ý:")
                    for _, rec_product in recommendations.iterrows():
                        st.write(f"- **{rec_product['ten_san_pham']}**")
                        st.write(f"  _{rec_product['mo_ta'][:50000]}..._")
                else:
                    st.write("Không có sản phẩm nào được đề xuất.")
            else:
                st.error("Không tìm thấy thông tin khách hàng.")
    
elif choice == 'Gợi ý sản phẩm theo thông tin sản phẩm':
    st.subheader("Gợi ý sản phẩm theo thông tin sản phẩm")
    # Nhập tên sản phẩm, tìm kiếm mã sản phẩm, và đề xuất các sản phẩm liên quan
    def get_products_recommendations(products, product_id, cosine_sim, nums=5):
        # Tìm chỉ mục sản phẩm dựa trên mã sản phẩm
        matching_indices = products.index[products['ma_san_pham'] == product_id].tolist()

        if not matching_indices:
            return pd.DataFrame()
        idx = matching_indices[0]

        # Tính toán độ tương đồng của sản phẩm được chọn với các sản phẩm khác
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sắp xếp sản phẩm theo độ tương đồng
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Lấy các sản phẩm tương tự (bỏ qua sản phẩm chính)
        sim_scores = sim_scores[1:nums + 1]

        # Lấy chỉ số sản phẩm
        product_indices = [i[0] for i in sim_scores]

        # Trả về danh sách sản phẩm được đề xuất
        return products.iloc[product_indices]

    # Đọc dữ liệu sản phẩm
    products = pd.read_csv('San_pham.csv')

    # Open and read file to cosine_sim_new
    with open('products_cosine_sim.pkl', 'rb') as f:
        cosine_sim_new = pickle.load(f)

    # Giao diện Streamlit
    st.title("Hệ thống gợi ý sản phẩm theo thông tin sản phẩm")

    st.image('hasaki_banner.jpg', use_container_width=True)

    # Người dùng nhập tên sản phẩm
    st.markdown('<p style="font-size:30px; font-weight:bold;">Nhập tên sản phẩm để tìm kiếm:</p>', unsafe_allow_html=True)
    product_name_input = st.text_input("product_name_input", key="product_name_input", label_visibility="hidden")

    # Kiểm tra tên sản phẩm
    if product_name_input:
        matching_products = products[products['ten_san_pham'].str.contains(product_name_input, case=False, na=False)]
        
        if not matching_products.empty:
            # Hiển thị các sản phẩm phù hợp với tên đã nhập
            st.write("### Sản phẩm tìm được:")
            for idx, product in matching_products.iterrows():
                st.write(f"- **{product['ten_san_pham']}** (Mã: {product['ma_san_pham']})")

            # Người dùng chọn sản phẩm từ danh sách
            selected_product = st.selectbox(
                "Chọn sản phẩm để xem gợi ý:",
                options=matching_products.itertuples(),
                format_func=lambda x: x.ten_san_pham
            )

            if selected_product:
                st.write("### Bạn đã chọn:")
                st.write(f"- **Tên:** {selected_product.ten_san_pham}")
                st.write(f"- **Mô tả:** {selected_product.mo_ta[:50000]}...")

                # Lấy danh sách sản phẩm gợi ý
                recommendations = get_products_recommendations(
                    products, selected_product.ma_san_pham, cosine_sim_new, nums=3
                )

                if not recommendations.empty:
                    st.write("### Các sản phẩm liên quan:")
                    for _, rec_product in recommendations.iterrows():
                        st.write(f"- **Tên:** {rec_product['ten_san_pham']}")
                        st.write(f"- **Mô tả:** {rec_product['mo_ta'][:50000]}...")
                else:
                    st.write("Không tìm thấy sản phẩm liên quan.")
        else:
            st.write("Không tìm thấy sản phẩm phù hợp.")
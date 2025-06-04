import streamlit as st
import pandas as pd
import numpy as np
import random
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import requests
import time
import streamlit.components.v1 as components

# Seed 고정
random.seed(42)
np.random.seed(42)

# 상품 리스트
product_pool = [f"P{str(i).zfill(3)}" for i in range(1, 21)]

# 주소 → 좌표 변환 함수
@st.cache_data
def get_coordinates(address):
    """
    카카오 지도 API를 사용하여 주소를 위도/경도로 변환
    캐시 처리로 중복 요청 방지
    """
    try:
        # 카카오 REST API 키 설정
        KAKAO_API_KEY = st.secrets["kakao_api_key"]
        
        # 카카오 지도 API 호출
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
        params = {"query": address}
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            result = response.json()
            if result["documents"]:
                # 첫 번째 검색 결과 사용
                first_result = result["documents"][0]
                lat = float(first_result["y"])  # 위도
                lon = float(first_result["x"])  # 경도
                return lat, lon, True
                
        return None, None, False
        
    except Exception as e:
        st.error(f"주소 검색 중 오류가 발생했습니다: {str(e)}")
        return None, None, False

# 거리 계산 함수
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# 매장 데이터 생성
def generate_store_master(n_stores=20, n_products=20):
    stores = []
    for i in range(n_stores):
        lat = round(np.random.uniform(37.45, 37.55), 6)
        lon = round(np.random.uniform(126.90, 127.10), 6)
        store_id = f"S{i+1:03}"
        sampled_products = random.sample(product_pool, k=random.randint(5, n_products))
        for product_id in sampled_products:
            stores.append({
                'store_id': store_id,
                'store_lat': lat,
                'store_lon': lon,
                'product_id': product_id,
                'inventory': np.random.randint(1, 6),
                'avg_shipping_hours': round(np.random.uniform(2.5, 8.0), 1),
                'penalty_count': np.random.randint(0, 4),
                'today_shipping_count': np.random.randint(0, 11)
            })
    return pd.DataFrame(stores)

# 주문 데이터 생성
def generate_orders(n_orders=1000, store_df=None):
    orders = []
    for i in range(n_orders):
        product = random.choice(product_pool)
        lat = round(np.random.uniform(37.45, 37.55), 6)
        lon = round(np.random.uniform(126.90, 127.10), 6)
        order_id = f"O{i+1:04}"
        possible_stores = store_df[store_df['product_id'] == product]
        actual_store = possible_stores.sample(1).iloc[0]['store_id'] if not possible_stores.empty else None
        orders.append({
            'order_id': order_id,
            'customer_lat': lat,
            'customer_lon': lon,
            'product_id': product,
            'actual_store_id': actual_store
        })
    return pd.DataFrame(orders)

# 응답 라벨 생성
def simulate_response_label(row):
    if row['distance_km'] <= 5 and row['inventory'] >= 3 and row['penalty_count'] == 0:
        return 1 if np.random.rand() < 0.95 else 0
    
    if row['distance_km'] > 12 or row['inventory'] <= 1 or row['penalty_count'] >= 3:
        return 0 if np.random.rand() < 0.95 else 1

    score = 0
    if row['distance_km'] <= 7:
        score += 0.3
    if row['inventory'] >= 3:
        score += 0.3
    if row['penalty_count'] == 0:
        score += 0.2
    if row['avg_shipping_hours'] < 4.0:
        score += 0.2

    return 1 if np.random.rand() < score else 0

# 출고 라벨 생성
def simulate_shipped_label(row):
    if row['responded'] == 0:
        return 0
    prob = 0.5
    if row['distance_km'] < 7:
        prob += 0.2
    if row['inventory'] >= 3:
        prob += 0.1
    if row['penalty_count'] == 0:
        prob += 0.15
    if row['avg_shipping_hours'] < 3.5:
        prob += 0.1
    prob = min(max(prob, 0.01), 0.99)
    return np.random.choice([0, 1], p=[1 - prob, prob])

# 입찰 로그 생성
def generate_bid_log(orders_df, store_df, max_candidates=5):
    bid_logs = []
    
    for _, order in orders_df.iterrows():
        candidates = store_df[store_df['product_id'] == order['product_id']]
        candidates = candidates.sample(n=min(max_candidates, len(candidates)))
        
        for _, row in candidates.iterrows():
            distance = haversine(
                order['customer_lat'], order['customer_lon'],
                row['store_lat'], row['store_lon']
            )
            bid_logs.append({
                'order_id': order['order_id'],
                'product_id': order['product_id'],
                'store_id': row['store_id'],
                'distance_km': round(distance, 2),
                'inventory': row['inventory'],
                'avg_shipping_hours': row['avg_shipping_hours'],
                'penalty_count': row['penalty_count'],
                'today_shipping_count': row['today_shipping_count']
            })
    
    bid_df = pd.DataFrame(bid_logs)
    bid_df['responded'] = bid_df.apply(simulate_response_label, axis=1)
    bid_df['shipped'] = bid_df.apply(simulate_shipped_label, axis=1)
    
    return bid_df

# 매장 추천 함수
def recommend_top_n_stores(customer_lat, customer_lon, product_id, store_df, model_resp, model_ship, top_n=3):
    available_stores = store_df[
        (store_df['product_id'] == product_id) & 
        (store_df['inventory'] > 0)
    ].copy()
    
    if len(available_stores) == 0:
        return pd.DataFrame()
    
    # 거리 계산
    available_stores['distance_km'] = available_stores.apply(
        lambda row: haversine(customer_lat, customer_lon, row['store_lat'], row['store_lon']),
        axis=1
    )
    
    # feature 구성
    features = ['distance_km', 'inventory', 'avg_shipping_hours', 'penalty_count', 'today_shipping_count']
    X = available_stores[features]
    
    # 응답 확률 예측
    available_stores['response_prob'] = model_resp.predict_proba(X)[:, 1]
    
    # 출고 확률 예측 (응답한 매장 가정 하에)
    available_stores['shipping_prob'] = model_ship.predict_proba(X)[:, 1]
    
    # 정렬 기준은 응답 확률로 (원하면 shipping_prob 기준도 가능)
    recommended_stores = available_stores.nlargest(top_n, 'response_prob')[
        ['store_id', 'store_lat', 'store_lon', 'response_prob', 'shipping_prob', 'distance_km', 'inventory']
    ]
    
    return recommended_stores


# Streamlit UI
st.set_page_config(page_title="O2O 매장 추천 시스템", layout="wide")
st.title("🏪 O2O 매장 추천 시스템")

# 데이터 및 모델 준비
@st.cache_data
def prepare_data_and_model():
    # 데이터 생성
    store_df = generate_store_master()
    orders_df = generate_orders(n_orders=3000, store_df=store_df)
    bid_df = generate_bid_log(orders_df, store_df)
    
    # 모델 feature 목록
    features = [
        'distance_km', 'inventory', 'avg_shipping_hours',
        'penalty_count', 'today_shipping_count',
    ]
    
    ##### [응답 예측 모델] #####
    X_resp = bid_df[features]
    y_resp = bid_df['responded']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_resp, y_resp, test_size=0.2, random_state=42
    )
    
    model_resp = XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )
    model_resp.fit(X_train_r, y_train_r)
    
    y_pred_r = model_resp.predict(X_test_r)
    model_performance = classification_report(y_test_r, y_pred_r)

    ##### [출고 예측 모델] #####
    bid_df_shipped = bid_df[bid_df['responded'] == 1]
    X_ship = bid_df_shipped[features]
    y_ship = bid_df_shipped['shipped']
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_ship, y_ship, test_size=0.2, random_state=42
    )
    
    model_ship = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_ship.fit(X_train_s, y_train_s)
    y_pred_s = model_ship.predict(X_test_s)
    ship_model_performance = classification_report(y_test_s, y_pred_s)

    return store_df, model_resp, model_performance, model_ship, ship_model_performance


store_df, model_resp, model_perf_resp, model_ship, model_perf_ship = prepare_data_and_model()


# 입력 폼
with st.form("recommendation_form"):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        product_id = st.selectbox(
            "상품 코드",
            options=sorted(store_df['product_id'].unique())
        )
    
    with col2:
        location_input_type = st.radio(
            "위치 입력 방식",
            options=["주소 검색", "좌표 직접 입력"],
            horizontal=True
        )
    
    if location_input_type == "주소 검색":
        address = st.text_input(
            "고객 주소 입력",
            placeholder="예: 서울시 강남구 테헤란로 8길 8",
            help="상세한 주소를 입력할수록 정확한 위치를 찾을 수 있습니다."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            customer_lat = st.number_input(
                "고객 위도",
                min_value=37.45,
                max_value=37.55,
                value=37.50,
                step=0.001,
                format="%.6f"
            )
        with col2:
            customer_lon = st.number_input(
                "고객 경도",
                min_value=126.90,
                max_value=127.10,
                value=127.00,
                step=0.001,
                format="%.6f"
            )
    
    submit_button = st.form_submit_button("Top 3 매장 추천")

# 추천 결과 표시
if submit_button:
    # 주소 입력 방식일 경우 좌표 변환
    if location_input_type == "주소 검색":
        if not address:
            st.error("주소를 입력해주세요.")
            st.stop()
        
        with st.spinner("주소를 검색중입니다..."):
            customer_lat, customer_lon, success = get_coordinates(address)
            
            if not success:
                st.error("주소를 찾을 수 없습니다. 더 자세한 주소를 입력하거나, 좌표를 직접 입력해주세요.")
                st.stop()
            else:
                st.success(f"📍 검색된 좌표: {customer_lat:.6f}, {customer_lon:.6f}")
    
    recommended_stores = recommend_top_n_stores(
        customer_lat, customer_lon, product_id,
        store_df, model_resp, model_ship
    )

    if len(recommended_stores) > 0:
        st.subheader("📊 추천 매장 목록")
        
        # 데이터프레임 표시
        formatted_df = recommended_stores.copy()
        
        # 추가 정보 가져오기
        store_info = store_df[store_df['store_id'].isin(formatted_df['store_id'])].copy()
        store_info = store_info.groupby('store_id').agg({
            'avg_shipping_hours': 'first',
            'penalty_count': 'first',
            'today_shipping_count': 'first'
        }).reset_index()
        
        # 추가 정보 병합
        formatted_df = formatted_df.merge(store_info, on='store_id', how='left')
        
        # 데이터 포맷팅
        formatted_df['response_prob'] = formatted_df['response_prob'].apply(lambda x: f"{x:.1%}")
        formatted_df['shipping_prob'] = formatted_df['shipping_prob'].apply(lambda x: f"{x:.1%}")
        formatted_df['distance_km'] = formatted_df['distance_km'].apply(lambda x: f"{x:.1f}km")
        formatted_df['avg_shipping_hours'] = formatted_df['avg_shipping_hours'].apply(lambda x: f"{x:.1f}시간")
        
        # 컬럼 순서 변경 및 이름 변경
        formatted_df = formatted_df[[
            'store_id', 'store_lat', 'store_lon', 'distance_km', 'inventory',
            'avg_shipping_hours', 'penalty_count', 'today_shipping_count',
            'response_prob', 'shipping_prob'
        ]]
        
        formatted_df.columns = [
            '매장 ID', '위도', '경도', '거리', '보유 재고',
            '평균 배송 시간', '누적 패널티 수', '당일 출고 처리량',
            '응답 확률', '출고 확률'
        ]
        
        st.dataframe(
            formatted_df,
            hide_index=True,
            use_container_width=True
        )
        
        # 지도 표시
        st.subheader("📍 위치 정보")
        
        # 카카오 맵 API 키 가져오기
        KAKAO_MAP_API_KEY = st.secrets["kakao_map_api_key"]
        
        # 고객 위치와 매장 위치 데이터 준비
        customer_location = {
            "lat": customer_lat,
            "lon": customer_lon
        }
        
        store_locations = []
        for _, store in recommended_stores.iterrows():
            store_locations.append({
                "store_id": store['store_id'],
                "lat": store['store_lat'],
                "lon": store['store_lon']
            })
        
        # HTML 템플릿 생성
        html_content = f"""
        <div id="map" style="width:100%; height:600px;"></div>
    <script type="text/javascript" src="https://dapi.kakao.com/v2/maps/sdk.js?appkey={KAKAO_MAP_API_KEY}&autoload=false&secure=true"></script>
    <script>
        kakao.maps.load(function() {{
            var container = document.getElementById('map');
            var options = {{
                center: new kakao.maps.LatLng({customer_lat}, {customer_lon}),
                level: 5
            }};
            var map = new kakao.maps.Map(container, options);

            // 고객 마커
            var customerMarker = new kakao.maps.Marker({{
                position: new kakao.maps.LatLng({customer_lat}, {customer_lon}),
                title: '고객 위치'
            }});
            customerMarker.setMap(map);

            // 매장 마커들
            """
        for store in store_locations:
            html_content += f"""
                    var marker = new kakao.maps.Marker({{
                        position: new kakao.maps.LatLng({store['lat']}, {store['lon']}),
                        title: '{store['store_id']}'
                    }});
                    marker.setMap(map);
            """

        html_content += """
                });
            </script>
        """

        # f"""
        #     <div id="map" style="width:100%;height:400px;"></div>
        #     <script type="text/javascript" src="https://dapi.kakao.com/v2/maps/sdk.js?appkey={KAKAO_MAP_API_KEY}"></script>
        #     <script>
        #         var container = document.getElementById('map');
        #         var options = {{
        #             center: new kakao.maps.LatLng({customer_lat}, {customer_lon}),
        #             level: 5
        #         }};
        #         var map = new kakao.maps.Map(container, options);
                
        #         // 고객 위치 마커 (파란색)
        #         var customerMarkerImage = new kakao.maps.MarkerImage(
        #             'https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/markerStar.png',
        #             new kakao.maps.Size(24, 35),
        #             new kakao.maps.Point(12, 35)
        #         );
                
        #         var customerMarker = new kakao.maps.Marker({{
        #             position: new kakao.maps.LatLng({customer_lat}, {customer_lon}),
        #             title: '고객 위치',
        #             image: customerMarkerImage
        #         }});
        #         customerMarker.setMap(map);
                
        #         // 매장 위치 마커들 (빨간색)
        #         var storePositions = [
        # """
        
        # # 매장 위치 데이터 추가
        # for store in store_locations:
        #     html_content += f"""
        #             {{
        #                 title: '{store["store_id"]}',
        #                 latlng: new kakao.maps.LatLng({store["lat"]}, {store["lon"]})
        #             }},
        #     """
        
        # html_content += """
        #         ];
                
        #         // 매장 마커 이미지
        #         var storeMarkerImage = new kakao.maps.MarkerImage(
        #             'https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/markerStar.png',
        #             new kakao.maps.Size(24, 35),
        #             new kakao.maps.Point(12, 35)
        #         );
                
        #         // 매장 마커들 생성
        #         storePositions.forEach(function(pos) {
        #             var marker = new kakao.maps.Marker({
        #                 map: map,
        #                 position: pos.latlng,
        #                 title: pos.title,
        #                 image: storeMarkerImage
        #             });
        #         });
        #     </script>
        # """
        
        # Streamlit에 HTML 삽입
        components.html(html_content, height=600)
        
        # 범례 표시
        st.markdown("""
        <div style='display: flex; gap: 20px; margin-top: 10px;'>
            <div style='display: flex; align-items: center;'>
                <div style='width: 12px; height: 12px; background-color: #2196F3; border-radius: 50%; margin-right: 5px;'></div>
                <span>고객 위치</span>
            </div>
            <div style='display: flex; align-items: center;'>
                <div style='width: 12px; height: 12px; background-color: #F44336; border-radius: 50%; margin-right: 5px;'></div>
                <span>추천 매장</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.error("⚠️ 해당 상품의 재고가 있는 매장을 찾을 수 없습니다.")

# # 모델 성능 표시 섹션
# st.divider()  # 구분선 추가
# show_model_performance = st.checkbox("응답 예측 모델 성능 보기", value=False)

# # 모델 성능이 체크되었을 때만 표시
# if show_model_performance:
#     st.subheader("📊 응답 예측 모델 성능")
#     st.text(model_performance)


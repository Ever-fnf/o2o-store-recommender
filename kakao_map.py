import streamlit as st
import streamlit.components.v1 as components

def display_kakao_map(customer_location, store_locations):
    """
    카카오 지도를 표시하는 함수
    
    Args:
        customer_location (dict): 고객 위치 정보 (lat, lon)
        store_locations (list): 매장 위치 정보 리스트 [{'store_id': 'S001', 'lat': xx.xx, 'lon': yy.yy}, ...]
    """
    
    # 카카오 맵 API 키 가져오기
    KAKAO_MAP_API_KEY = st.secrets["kakao_map_api_key"]
    
    # HTML 템플릿 생성
    html_content = f"""
        <div id="map" style="width:100%;height:400px;"></div>
        <script type="text/javascript" src="https://dapi.kakao.com/v2/maps/sdk.js?appkey={KAKAO_MAP_API_KEY}"></script>
        <script>
            var container = document.getElementById('map');
            var options = {{
                center: new kakao.maps.LatLng({customer_location['lat']}, {customer_location['lon']}),
                level: 5
            }};
            var map = new kakao.maps.Map(container, options);
            
            // 고객 위치 마커 (파란색)
            var customerMarkerImage = new kakao.maps.MarkerImage(
                'https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/markerStar.png',
                new kakao.maps.Size(24, 35),
                new kakao.maps.Point(12, 35)
            );
            
            var customerMarker = new kakao.maps.Marker({{
                position: new kakao.maps.LatLng({customer_location['lat']}, {customer_location['lon']}),
                title: '고객 위치',
                image: customerMarkerImage
            }});
            customerMarker.setMap(map);
            
            // 매장 위치 마커들 (빨간색)
            var storePositions = [
    """
    
    # 매장 위치 데이터 추가
    for store in store_locations:
        html_content += f"""
                {{
                    title: '{store["store_id"]}',
                    latlng: new kakao.maps.LatLng({store["lat"]}, {store["lon"]})
                }},
        """
    
    html_content += """
            ];
            
            // 매장 마커 이미지
            var storeMarkerImage = new kakao.maps.MarkerImage(
                'https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/markerStar.png',
                new kakao.maps.Size(24, 35),
                new kakao.maps.Point(12, 35)
            );
            
            // 매장 마커들 생성
            storePositions.forEach(function(pos) {
                var marker = new kakao.maps.Marker({
                    map: map,
                    position: pos.latlng,
                    title: pos.title,
                    image: storeMarkerImage
                });
            });
        </script>
    """
    
    # Streamlit에 HTML 삽입
    components.html(html_content, height=400)

# 예시 데이터로 테스트
if __name__ == "__main__":
    st.title("🗺️ 추천 매장 위치")
    
    # 테스트용 데이터
    customer = {
        "lat": 37.4979,
        "lon": 127.0276
    }
    
    stores = [
        {"store_id": "S001", "lat": 37.5079, "lon": 127.0376},
        {"store_id": "S002", "lat": 37.4879, "lon": 127.0176},
        {"store_id": "S003", "lat": 37.5179, "lon": 127.0476}
    ]
    
    # 지도 표시
    display_kakao_map(customer, stores)
    
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
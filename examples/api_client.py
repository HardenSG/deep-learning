import requests
import json


class QuantAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/api/health")
        return response.json()
    
    def get_stock_list(self):
        response = requests.get(f"{self.base_url}/api/stocks")
        return response.json()
    
    def get_stock_data(self, stock_code, limit=100):
        response = requests.get(
            f"{self.base_url}/api/data/{stock_code}",
            params={"limit": limit}
        )
        return response.json()
    
    def update_data(self, stock_codes=None, days_back=30):
        data = {
            "stock_codes": stock_codes,
            "days_back": days_back
        }
        response = requests.post(
            f"{self.base_url}/api/data/update",
            json=data
        )
        return response.json()
    
    def predict(self, stock_code, days=5):
        data = {
            "stock_code": stock_code,
            "days": days
        }
        response = requests.post(
            f"{self.base_url}/api/predict",
            json=data
        )
        return response.json()
    
    def get_model_info(self, stock_code):
        response = requests.get(f"{self.base_url}/api/model/info/{stock_code}")
        return response.json()


def main():
    print("=" * 60)
    print("API客户端示例")
    print("=" * 60)
    
    client = QuantAPIClient()
    
    print("\n1. 健康检查...")
    health = client.health_check()
    print(f"   状态: {health}")
    
    print("\n2. 获取股票列表...")
    stocks = client.get_stock_list()
    print(f"   股票数量: {stocks['total']}")
    print(f"   前5只股票: {stocks['stocks'][:5]}")
    
    stock_code = "000001"
    
    print(f"\n3. 获取股票数据: {stock_code}")
    try:
        data = client.get_stock_data(stock_code, limit=10)
        print(f"   数据量: {data['total']}")
        print(f"   最新数据: {data['data'][-1] if data['data'] else 'None'}")
    except Exception as e:
        print(f"   错误: {e}")
    
    print(f"\n4. 预测股票走势: {stock_code}")
    try:
        prediction = client.predict(stock_code)
        print(f"   预测结果: {json.dumps(prediction, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"   错误: {e}")
    
    print(f"\n5. 获取模型信息: {stock_code}")
    try:
        model_info = client.get_model_info(stock_code)
        print(f"   模型信息: {json.dumps(model_info, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"   错误: {e}")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

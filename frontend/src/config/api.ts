/**
 * API 配置文件
 * 
 * 自动检测当前主机地址，确保从任何位置访问都能正确连接后端
 */

/**
 * 获取后端 API 基础地址
 * 自动检测当前访问的主机，确保前后端在同一服务器上时可以正确连接
 */
function getApiBaseUrl(): string {
  // 如果在浏览器环境中，使用当前访问的主机
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    // 使用当前主机名和端口 8002
    return `http://${hostname}:8002`;
  }
  
  // 非浏览器环境（SSR等），使用默认值
  return 'http://127.0.0.1:8002';
}

export const API_CONFIG = {
  /**
   * 后端 API 基础地址
   * 
   * 自动检测当前访问的主机地址，确保从任何位置访问都能正确连接后端
   * 例如：
   * - 从 http://192.168.12.132:3000 访问时，自动使用 http://192.168.12.132:8002
   * - 从 http://localhost:3000 访问时，自动使用 http://localhost:8002
   * - 从 http://127.0.0.1:3000 访问时，自动使用 http://127.0.0.1:8002
   */
  BASE_URL: getApiBaseUrl(),
};

export default API_CONFIG;

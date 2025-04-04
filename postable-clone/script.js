document.addEventListener('DOMContentLoaded', () => {
    // 获取需要的元素
    const logoButton = document.querySelector('.logo-button');
    const contentWrapper = document.querySelector('.content-wrapper');
    const searchContainer = document.querySelector('.search-container');
    const searchInput = document.querySelector('.search-input');
    const gradientBanner = document.querySelector('.gradient-banner');
    const buttonText = document.querySelector('.button-text');
    const searchIcon = document.querySelector('.search-icon');
    
    // 设置初始样式和宽度
    searchContainer.style.position = 'fixed';

    // 设置一个固定的初始宽度，不再动态计算
    searchContainer.style.width = '180px';
    
    // 添加点击事件
    logoButton.addEventListener('click', toggleSearchState);
    
    // 切换搜索状态的函数
    function toggleSearchState() {
        const isActive = contentWrapper.classList.contains('active-state');
        
        if (!isActive) {
            // 展开搜索框
            expandSearch();
        } else {
            // 收起搜索框
            collapseSearch();
        }
    }
    
    // 展开搜索框
    function expandSearch() {
        // 强制布局重绘确保动画起始正确
        searchContainer.style.transition = 'none';
        // 触发重排
        searchContainer.offsetWidth;
        searchContainer.style.transition = 'all 0.7s cubic-bezier(0.33, 1, 0.68, 1)';
        
        // 添加活跃状态前确保初始宽度正确
        contentWrapper.classList.add('active-state');
        
        // 确保搜索框变宽
        searchContainer.style.width = '600px';
        
        // 禁止滚动
        document.body.style.overflow = 'hidden';
        
        // 延迟聚焦输入框并确保搜索图标可见
        setTimeout(() => {
            searchInput.focus();
            searchIcon.style.opacity = '1';
            searchInput.style.opacity = '1';
            searchInput.style.width = 'calc(100% - 80px)';
        }, 300);
    }
    
    // 收起搜索框
    function collapseSearch() {
        // 移除活跃状态
        contentWrapper.classList.remove('active-state');
        
        // 恢复滚动
        document.body.style.overflow = '';
        
        // 重置输入框并取消焦点
        searchInput.blur();
        
        // 确保搜索图标隐藏
        searchIcon.style.opacity = '0';
        searchInput.style.opacity = '0';
        searchInput.style.width = '0';
        
        // 确保容器回到原始宽度
        setTimeout(() => {
            searchContainer.style.width = '180px';
        }, 100);
    }
    
    // 点击外部区域关闭
    document.addEventListener('click', (event) => {
        const isActive = contentWrapper.classList.contains('active-state');
        if (isActive && 
            !searchContainer.contains(event.target) && 
            !gradientBanner.contains(event.target)) {
            
            collapseSearch();
        }
    });
    
    // 阻止事件冒泡
    searchContainer.addEventListener('click', (event) => {
        event.stopPropagation();
    });
    
    gradientBanner.addEventListener('click', (event) => {
        event.stopPropagation();
    });
    
    // 搜索功能
    searchInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            const query = searchInput.value.trim();
            if (query) {
                alert(`搜索: ${query}`);
            }
        }
    });
    
    // 确保始终显示在上层
    window.addEventListener('scroll', () => {
        searchContainer.style.position = 'fixed';
        searchContainer.style.zIndex = '1000';
    });
});

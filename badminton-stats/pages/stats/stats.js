// 统计与视频复盘页
Page({
  data: {
    // 比赛数据
    match: null,
    hasLogs: false,

    // 选手名称
    playerAName: '我方',
    playerBName: '对方',

    // 四维技术统计（A 和 B 分别计数）
    categories: [
      { key: 'smash_score', label: '杀球得分' },
      { key: 'net_score',   label: '网前得分' },
      { key: 'out_error',   label: '出界失误' },
      { key: 'net_error',   label: '下网失误' }
    ],

    // 选手统计数据
    statsA: { smash_score: 0, net_score: 0, out_error: 0, net_error: 0 },
    statsB: { smash_score: 0, net_score: 0, out_error: 0, net_error: 0 },

    // 柱状图缩放基准（所有条的最大值，用于百分比宽度）
    maxStatValue: 1,

    // ===== 视频播放相关 =====
    filters: [
      { key: 'all',         label: '全部集锦', icon: '🎬' },
      { key: 'smash_score', label: '杀球得分', icon: '💥' },
      { key: 'net_score',   label: '网前得分', icon: '🥅' },
      { key: 'out_error',   label: '出界失误', icon: '📤' },
      { key: 'net_error',   label: '下网失误', icon: '📥' }
    ],

    selectedFilter: '',
    filterLabel: '',

    // 筛选后的日志列表
    filteredLogs: [],
    currentVideoIndex: 0,
    videoSrc: '',
    currentLog: null,

    // 播放状态
    isPlaying: false,
    playbackComplete: false,
    videoHidden: true,

    // 视频总数 & 当前序号（用于显示 "3 / 12"）
    totalClips: 0,
    currentClipNo: 0
  },

  /* ========== 生命周期 ========== */
  onLoad: function (options) {
    const app = getApp();

    // 优先读取 currentMatch（新数据模型），兼容旧的 matches[index]
    let match = app.globalData.currentMatch || null;
    if (!match && options.index !== undefined) {
      const idx = parseInt(options.index);
      match = app.globalData.matches ? app.globalData.matches[idx] : null;
    }

    if (!match) {
      wx.showToast({ title: '未找到比赛数据', icon: 'error' });
      return;
    }

    this.setData({ match: match });
    this.computeStats(match);
  },

  onReady: function () {
    this.videoContext = wx.createVideoContext('statsVideo');
  },

  /* ========== 数据统计 ========== */

  /**
   * 从 match.logs 中汇总双方技术统计
   * 每条 log 结构：{ type, player, videoPath, scoreA, scoreB, timestamp }
   */
  computeStats: function (match) {
    const logs = match.logs || [];

    if (logs.length === 0) {
      this.setData({ hasLogs: false });
      return;
    }

    const statsA = { smash_score: 0, net_score: 0, out_error: 0, net_error: 0 };
    const statsB = { smash_score: 0, net_score: 0, out_error: 0, net_error: 0 };

    logs.forEach(log => {
      const type = log.type;
      if (log.player === 'A' && statsA.hasOwnProperty(type)) {
        statsA[type]++;
      } else if (log.player === 'B' && statsB.hasOwnProperty(type)) {
        statsB[type]++;
      }
    });

    // 计算所有值中的最大值，用于柱状图比例缩放
    const allValues = [
      ...Object.values(statsA),
      ...Object.values(statsB)
    ];
    const maxVal = Math.max(...allValues, 1);

    this.setData({
      hasLogs: true,
      statsA: statsA,
      statsB: statsB,
      maxStatValue: maxVal,
      // 同时取选手名
      playerAName: match.playerA || '我方',
      playerBName: match.playerB || '对方'
    });
  },

  /* ========== 视频筛选 ========== */

  /**
   * 点击筛选标签 → 筛选 logs → 开始连续播放
   */
  onFilterTap: function (e) {
    const filterKey = e.currentTarget.dataset.key;
    const filterItem = this.data.filters.find(f => f.key === filterKey);
    const filterLabel = filterItem ? filterItem.label : filterKey;

    const logs = this.data.match.logs || [];

    // 按类型筛选（"all" 表示不筛选）
    let filtered;
    if (filterKey === 'all') {
      filtered = logs.filter(l => l.videoPath); // 只要有视频
    } else {
      filtered = logs.filter(l => l.type === filterKey && l.videoPath);
    }

    if (filtered.length === 0) {
      wx.showToast({
        title: '该分类暂无视频片段',
        icon: 'none',
        duration: 2000
      });
      return;
    }

    this.setData({
      selectedFilter: filterKey,
      filterLabel: filterLabel,
      filteredLogs: filtered,
      totalClips: filtered.length,
      currentVideoIndex: 0,
      currentClipNo: 1,
      videoSrc: filtered[0].videoPath,
      currentLog: filtered[0],
      isPlaying: true,
      playbackComplete: false,
      videoHidden: false
    });
  },

  /* ========== 视频连续播放 ========== */

  /**
   * 当前视频播放完毕 → 自动播放下一条
   */
  onVideoEnded: function () {
    const { currentVideoIndex, filteredLogs } = this.data;
    const nextIndex = currentVideoIndex + 1;

    if (nextIndex >= filteredLogs.length) {
      // 全部播完
      this.setData({
        isPlaying: false,
        playbackComplete: true
      });
      wx.showToast({ title: '已播完所有片段', icon: 'none' });
      return;
    }

    // 切换到下一条
    const nextLog = filteredLogs[nextIndex];
    this.setData({
      currentVideoIndex: nextIndex,
      currentClipNo: nextIndex + 1,
      videoSrc: nextLog.videoPath,
      currentLog: nextLog
    });

    // 确保自动播放（部分机型 setData 后 autoplay 不生效）
    setTimeout(() => {
      if (this.videoContext) {
        this.videoContext.play();
      }
    }, 150);
  },

  /**
   * 视频加载失败 → 跳过并尝试下一条
   */
  onVideoError: function (e) {
    console.error('视频加载失败:', e.detail.errMsg);
    wx.showToast({ title: '视频加载失败，跳过', icon: 'none', duration: 1500 });
    // 跳过当前，继续播放下一条
    this.onVideoEnded();
  },

  /**
   * 上一段 / 下一段 手动切换
   */
  onPrevClip: function () {
    const { currentVideoIndex, filteredLogs } = this.data;
    if (currentVideoIndex <= 0) return;
    const prevIndex = currentVideoIndex - 1;
    const prevLog = filteredLogs[prevIndex];
    this.setData({
      currentVideoIndex: prevIndex,
      currentClipNo: prevIndex + 1,
      videoSrc: prevLog.videoPath,
      currentLog: prevLog,
      playbackComplete: false
    });
    setTimeout(() => {
      if (this.videoContext) this.videoContext.play();
    }, 150);
  },

  onNextClip: function () {
    const { currentVideoIndex, filteredLogs } = this.data;
    if (currentVideoIndex >= filteredLogs.length - 1) return;
    const nextIndex = currentVideoIndex + 1;
    const nextLog = filteredLogs[nextIndex];
    this.setData({
      currentVideoIndex: nextIndex,
      currentClipNo: nextIndex + 1,
      videoSrc: nextLog.videoPath,
      currentLog: nextLog,
      playbackComplete: false
    });
    setTimeout(() => {
      if (this.videoContext) this.videoContext.play();
    }, 150);
  },

  /**
   * 从列表中点击某一条直接跳播
   */
  onJumpToClip: function (e) {
    const idx = e.currentTarget.dataset.index;
    const log = this.data.filteredLogs[idx];
    this.setData({
      currentVideoIndex: idx,
      currentClipNo: idx + 1,
      videoSrc: log.videoPath,
      currentLog: log,
      playbackComplete: false,
      isPlaying: true
    });
    setTimeout(() => {
      if (this.videoContext) this.videoContext.play();
    }, 150);
  },

  /**
   * 关闭视频播放器
   */
  onCloseVideo: function () {
    if (this.videoContext) this.videoContext.pause();
    this.setData({
      isPlaying: false,
      videoHidden: true,
      videoSrc: '',
      playbackComplete: false
    });
  },

  /**
   * 回到筛选列表（视频关闭后重新选择）
   */
  onBackToFilter: function () {
    if (this.videoContext) this.videoContext.pause();
    this.setData({
      isPlaying: false,
      videoHidden: true,
      videoSrc: '',
      playbackComplete: false,
      selectedFilter: '',
      filterLabel: '',
      filteredLogs: [],
      currentVideoIndex: 0
    });
  }
});

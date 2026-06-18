// 首页 - 比赛列表与入口
Page({
  data: {
    matches: []
  },

  onShow: function () {
    const app = getApp();
    this.setData({ matches: app.globalData.matches });
  },

  // 新建比赛
  startNewMatch: function () {
    wx.navigateTo({ url: '/pages/match/match' });
  },

  // 查看统计
  viewStats: function (e) {
    const idx = e.currentTarget.dataset.index;
    wx.navigateTo({ url: '/pages/stats/stats?index=' + idx });
  }
});

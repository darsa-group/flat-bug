import cv2
import numpy as np
import morphsnakes as ms
import matplotlib.pyplot as plt

from skimage.filters import gaussian
from skimage.segmentation import active_contour




def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.

    Parameters
    ----------
    background : (M, N) array
        Image to be plotted as the background of the visual evolution.
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.

    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.

    """

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback

# here, resize image

def refine_contour(img, ct, size=400):

    # Initialization of the level-set.
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(grey)

    h,w = grey.shape
    grey = cv2.resize(grey, (size, size))
    gimg = grey/255.0


    mask = cv2.drawContours(mask, [ct], -1, (255, ), -1, cv2.LINE_8)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    # todo, scale image down!
    # init_ls = ms.circle_level_set(img.shape, (163, 137), 135)
    # print(init_ls)

    # Callback for visual plotting
    # callback = visual_callback_2d(grey)

    # MorphGAC.
    o = ms.morphological_chan_vese(gimg, iterations=100,
                                             init_level_set=mask,
                                               smoothing=1, lambda1=.4, lambda2=5,
                                   # iter_callback= callback
                                               )
    # todo, scale new mask UP!
    o = cv2.resize(o, (w, h), interpolation=cv2.INTER_NEAREST)
    assert  o.shape == (h,w)
    cts,_ = cv2.findContours(o.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    largest = None
    for c in cts:
        a = cv2.contourArea(c)
        if a > largest_area:
            largest = c
            largest_area = a
    return largest


img = cv2.imread("images/mask_refinement/cropped.jpg")
mask = cv2.imread("images/mask_refinement/cropped-mask.png", cv2.IMREAD_GRAYSCALE)
cts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ct = cts[0]
new_ct = refine_contour(img, ct)

display = np.copy(img)
# print(ct)
cv2.drawContours(display, [ct], -1,  (255, 0 ,0), 2, cv2.LINE_AA)
cv2.drawContours(display, [new_ct], -1,  (0, 0 ,255), 2, cv2.LINE_AA)
cv2.imshow("test", display)
cv2.waitKey(-1)





#
# # s = np.linspace(0, 2*np.pi, 400)
# # r = 100 + 100*np.sin(s)
# # c = 220 + 100*np.cos(s)
# # init = np.array([r, c]).T
# # print(init)
# img = img.T
# init = np.vstack(cts).squeeze().astype(np.float)
#
# snake = active_contour(gaussian(img, 1, preserve_range=False),
#                        init, alpha=0.1, beta=10.0, gamma=0.5)
#
# fig, ax = plt.subplots(figsize=(7, 7))
# ax.imshow(img, cmap=plt.cm.gray)
# ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
# ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])
#
# plt.show()
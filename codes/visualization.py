
# import modules
# from google.colab import auth # only for google colab
# auth.authenticate_user() # # only for google colab
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams.update({'font.size': 14}) # set the font size globally
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_xarray(data, at_time, label, cmap='jet', save_as = None, show=False, vmin=None, vmax=None, extent=None, projection=ccrs.PlateCarree()):

    # preprocess the xarray data
    data_plot = data.transpose('time', 'latitude', 'longitude')
    data_plot = data_plot.assign_coords(longitude=(((data_plot.longitude))))

    # pick the state at the specified time step
    data_plot = data_plot.isel(time=at_time)

    # plot the data
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=projection)
    data_plot.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs={'label': label})
    ax.coastlines() # add coastlines
    ax.add_feature(cfeature.BORDERS) # add borders
    # ax.gridlines(draw_labels=True) # add gridlines
    # set extent if provided
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    plt.title(f"{label} of {data.time.values[at_time]}") # set title
    plt.tight_layout()

    # Save
    if save_as:
            plt.savefig(save_as)

    if show==True:
        plt.show()

    plt.close()

def plot_xarray_comparison(data, data_true, at_time, label, cmap='jet', save_as=None, show=False, vmin=None, vmax=None, projection=ccrs.PlateCarree()):
    # preprocess the xarray data
    data_plot = data.transpose('time', 'latitude', 'longitude')
    data_plot = data_plot.assign_coords(longitude=(((data_plot.longitude))))
    data_true_plot = data_true.transpose('time', 'latitude', 'longitude')
    data_true_plot = data_true_plot.assign_coords(longitude=(((data_true_plot.longitude))))

    # Initialize the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 8), subplot_kw={'projection': projection})
    
    # Plot data at the specified time step
    img1 = data_plot.isel(time=at_time).plot(ax=ax1, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    # cbar1 = fig.colorbar(img1, ax=ax1, label=label)
    ax1.coastlines()  # add coastlines
    ax1.add_feature(cfeature.BORDERS)  # add borders
    ax1.gridlines(draw_labels=True)  # add gridlines
    ax1.set_title(f"predicted {label} at {data.time.values[at_time]}")  # set title
    
    # Plot data_true at the specified time step
    img2 = data_true_plot.isel(time=at_time).plot(ax=ax2, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    # cbar2 = fig.colorbar(img2, ax=ax2, label=label)
    ax2.coastlines()  # add coastlines
    ax2.add_feature(cfeature.BORDERS)  # add borders
    ax2.gridlines(draw_labels=True)  # add gridlines
    ax2.set_title(f"true {label} at {data_true.time.values[at_time]}")  # set title

    # Add a single colorbar
    cbar = fig.colorbar(img1, ax=[ax1, ax2], orientation='horizontal', pad=0.1, aspect=50, shrink=0.8)
    cbar.set_label(label)

    # plt.tight_layout()

    # Save
    if save_as:
        plt.savefig(save_as)

    if show==True:
        plt.show()

    plt.close()

def plot_xarray_movie(data, start_t=0, end_t=-1, label="var", cmap='jet', interval=200, save_as='animation.gif', vmin=None, vmax=None, projection=ccrs.PlateCarree()):
    # preprocess the xarray data
    data_plot = data.transpose('time', 'latitude', 'longitude')
    data_plot = data_plot.assign_coords(longitude=(((data_plot.longitude))))

    # Initialize the plot
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=projection)
    img = data_plot.isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    cbar = fig.colorbar(img, ax=ax, label=label)  # Create colorbar only once
    ax.coastlines()  # add coastlines
    ax.add_feature(cfeature.BORDERS)  # add borders
    ax.gridlines(draw_labels=True)  # add gridlines
    title = ax.set_title(f"{label} of {data.time.values[0]}")  # set initial title

    # Update function for animation
    def update(frame):
        img.set_array(data_plot.isel(time=frame).values.flatten())
        title.set_text(f"{label} of {data.time.values[frame]}")

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(data_plot.time[start_t:end_t]), interval=interval)

    # Save the animation as a GIF
    ani.save(save_as, writer='pillow')

    plt.close()

def plot_xarray_movie_comparison(data, data_true, start_t, end_t, label, cmap='jet', interval=200, save_as='comparison_animation.gif', vmin=None, vmax=None, projection=ccrs.PlateCarree()):
    # preprocess the xarray data
    data_plot = data.transpose('time', 'latitude', 'longitude')
    data_plot = data_plot.assign_coords(longitude=(((data_plot.longitude))))
    data_true_plot = data_true.transpose('time', 'latitude', 'longitude')
    data_true_plot = data_true_plot.assign_coords(longitude=(((data_true_plot.longitude))))

    # Initialize the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), subplot_kw={'projection': projection})
    
    # Plot data
    img1 = data_plot.isel(time=0).plot(ax=ax1, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    # cbar1 = fig.colorbar(img1, ax=ax1, label=label)
    ax1.coastlines()  # add coastlines
    ax1.add_feature(cfeature.BORDERS)  # add borders
    ax1.gridlines(draw_labels=True)  # add gridlines
    title1 = ax1.set_title(f"predicted {label} of {data.time.values[0]}")  # set initial title
    
    # Plot data_true
    img2 = data_true_plot.isel(time=0).plot(ax=ax2, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    # cbar2 = fig.colorbar(img2, ax=ax2, label=label)
    ax2.coastlines()  # add coastlines
    ax2.add_feature(cfeature.BORDERS)  # add borders
    ax2.gridlines(draw_labels=True)  # add gridlines
    title2 = ax2.set_title(f"true {label} of {data_true.time.values[0]}")  # set initial title

    # Add a single colorbar
    cbar = fig.colorbar(img1, ax=[ax1, ax2], orientation='horizontal', pad=0.1, aspect=50, shrink=0.8)
    cbar.set_label(label)

    # Update function for animation
    def update(frame):

        img1.set_array(data_plot.isel(time=frame).values.flatten())
        title1.set_text(f"predicted {label} of {data.time.values[frame]}")

        img2.set_array(data_true_plot.isel(time=frame).values.flatten())
        title2.set_text(f"true {label} of {data_true.time.values[frame]}")

        # # Update the color limits dynamically based on the data
        # current_vmin_1 = data_plot.isel(time=frame).values.min()
        # current_vmax_1 = data_plot.isel(time=frame).values.max()
        # current_vmin_2 = data_true_plot.isel(time=frame).values.min()
        # current_vmax_2 = data_true_plot.isel(time=frame).values.max()
        
        # # Update the color limits
        # img1.set_clim(current_vmin_1, current_vmax_1)
        # img2.set_clim(current_vmin_2, current_vmax_2)

        # # Update the colorbars
        # cbar1.update_normal(img1)
        # cbar2.update_normal(img2)

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(data_plot.time[start_t:end_t]), interval=interval)

    # Save the animation as a GIF
    if save_as:
        ani.save(save_as, writer='pillow')

    plt.close()
